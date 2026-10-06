"""Behaviour tests for the binary frame codec.

The tests only describe what a caller of the codec is allowed to observe:
the bytes that go out on the wire, the frames that come back, and the errors
that the parser reports.  Run them from the project root:

    python3 -m unittest discover -s tests -v
"""

import os
import struct
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import codec.core as codec


def raw_frame(msg_type, payload, version=codec.VERSION):
    """Assemble a frame straight from the wire format, checksum included."""
    body = (
        bytes([(version << 4) | msg_type])
        + codec.encode_uvarint(len(payload))
        + payload
    )
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


class RoundTripTests(unittest.TestCase):
    """Frames a peer sends in one piece."""

    def test_a_small_frame_round_trips(self):
        raw = codec.encode_frame(codec.MSG_ECHO, b"hello")
        self.assertEqual(len(raw), 1 + 1 + 5 + 4)
        frame, used = codec.decode_frame(raw)
        self.assertEqual(frame.version, codec.VERSION)
        self.assertEqual(frame.msg_type, codec.MSG_ECHO)
        self.assertEqual(frame.payload, b"hello")
        self.assertEqual(used, len(raw))
        parser = codec.FrameParser()
        parser.feed(raw)
        parsed = parser.read_frame()
        self.assertEqual(parsed.payload, b"hello")
        self.assertEqual(parser.pending(), 0)

    def test_length_prefixes_round_trip_and_stay_canonical(self):
        for value in (0, 1, 127, 128, 300, 16384, 1 << 20):
            wire = codec.encode_uvarint(value)
            got, used = codec.decode_uvarint(wire)
            self.assertEqual(used, len(wire))
            self.assertEqual(got, value)
        self.assertEqual(codec.encode_uvarint(300), b"\xac\x02")
        self.assertRaises(codec.FrameError, codec.decode_uvarint, b"\x80\x00")
        payload = bytes(range(200))
        raw = codec.encode_frame(codec.MSG_BLOB, payload)
        self.assertEqual(len(raw), 1 + 2 + len(payload) + 4)
        length, prefix = codec.decode_uvarint(raw, 1)
        self.assertEqual(prefix, 2)
        self.assertEqual(length, len(payload))
        frame, consumed = codec.decode_frame(raw)
        self.assertEqual(frame.payload, payload)
        self.assertEqual(consumed, len(raw))

    def test_a_payload_at_the_limit_is_accepted(self):
        payload = b"y" * codec.MAX_PAYLOAD
        raw = codec.encode_frame(codec.MSG_BLOB, payload)
        self.assertEqual(len(raw), 1 + 2 + codec.MAX_PAYLOAD + 4)
        length, prefix = codec.decode_uvarint(raw, 1)
        self.assertEqual(length, codec.MAX_PAYLOAD)
        frame, consumed = codec.decode_frame(raw)
        self.assertEqual(frame.payload, payload)
        self.assertEqual(consumed, len(raw))

    def test_signed_payloads_round_trip(self):
        for value in (-1, -64, 0, 1, 63, -7):
            wire = codec.encode_svarint(value)
            got, used = codec.decode_svarint(wire)
            self.assertEqual(used, len(wire))
            self.assertEqual(got, value)
        raw = codec.encode_int_frame(-2)
        frame, consumed = codec.decode_frame(raw)
        self.assertEqual(frame.msg_type, codec.MSG_INT)
        self.assertEqual(consumed, len(raw))
        self.assertEqual(codec.decode_int_frame(frame), -2)

    def test_frames_arrive_one_after_another_and_split_across_feeds(self):
        first = codec.encode_frame(codec.MSG_PING, b"")
        second = codec.encode_frame(codec.MSG_BLOB, b"second")
        parser = codec.FrameParser()
        parser.feed(first + second[:2])
        one = parser.read_frame()
        self.assertEqual(one.msg_type, codec.MSG_PING)
        self.assertEqual(parser.pending(), 2)
        self.assertIsNone(parser.read_frame())
        parser.feed(second[2:])
        two = parser.read_frame()
        self.assertEqual(two.payload, b"second")
        self.assertEqual(parser.pending(), 0)
        self.assertEqual(
            [item.payload for item in codec.decode_all(first + second)],
            [b"", b"second"],
        )


class ChecksumTests(unittest.TestCase):
    """The checksum has to cover the bytes that precede it."""

    def test_the_checksum_covers_the_header_and_the_length_prefix(self):
        raw = codec.encode_frame(codec.MSG_ECHO, b"hello")
        stored = struct.unpack("<I", raw[-4:])[0]
        self.assertEqual(stored, zlib.crc32(raw[:-4]) & 0xFFFFFFFF)
        changed = bytes([raw[0] ^ 0x01]) + raw[1:]
        self.assertRaises(codec.FrameError, codec.decode_frame, changed)


class FailureTests(unittest.TestCase):
    """Bytes that must be reported instead of being accepted or dropped."""

    def test_a_corrupt_payload_is_reported_and_kept_in_the_buffer(self):
        raw = bytearray(codec.encode_frame(codec.MSG_BLOB, b"payload"))
        raw[2] ^= 0x01
        raw = bytes(raw)
        self.assertRaises(codec.FrameError, codec.decode_frame, raw)
        parser = codec.FrameParser()
        parser.feed(raw)
        self.assertRaises(codec.FrameError, parser.read_frame)
        self.assertEqual(parser.pending(), len(raw))
        self.assertRaises(codec.FrameError, parser.read_frame)

    def test_a_buffer_that_ends_mid_frame_is_not_silently_dropped(self):
        first = codec.encode_frame(codec.MSG_PING, b"")
        second = codec.encode_frame(codec.MSG_BLOB, b"second")
        self.assertRaises(codec.FrameError, codec.decode_all, first + second[:3])
        self.assertEqual(
            [(item.msg_type, item.payload) for item in codec.decode_all(first + second)],
            [(codec.MSG_PING, b""), (codec.MSG_BLOB, b"second")],
        )

    def test_frames_with_a_foreign_version_are_rejected(self):
        raw = codec.encode_frame(codec.MSG_PING, b"", version=codec.VERSION + 1)
        self.assertEqual(raw[0] >> 4, codec.VERSION + 1)
        self.assertRaises(codec.FrameError, codec.decode_frame, raw)
        parser = codec.FrameParser()
        parser.feed(raw)
        self.assertRaises(codec.FrameError, parser.read_frame)
        self.assertEqual(parser.pending(), len(raw))

    def test_a_frame_above_the_payload_limit_is_rejected(self):
        payload = b"x" * (codec.MAX_PAYLOAD + 1)
        raw = raw_frame(codec.MSG_BLOB, payload)
        self.assertEqual(len(raw), 1 + 2 + len(payload) + 4)
        self.assertRaises(codec.FrameError, codec.decode_frame, raw)


if __name__ == "__main__":
    unittest.main()
