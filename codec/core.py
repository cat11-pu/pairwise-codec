"""Binary frame codec kernel built on the standard library only.

The wire format is a small self describing frame::

    +--------+----------------+---------+----------+
    | header | payload length | payload | checksum |
    | 1 byte | varint         | N bytes | 4 bytes  |
    +--------+----------------+---------+----------+

The header byte carries a four bit version number in its high nibble and a
four bit message type in its low nibble.  The payload length is an unsigned
varint, and the checksum is the CRC32 of the frame bytes that precede it,
stored little endian.  Signed integers travel as zigzag varints.

Everything stays in memory: callers hand bytes to :func:`decode_frame`,
:func:`decode_all` or a :class:`FrameParser` and get :class:`Frame` objects
back.
"""

import struct
import zlib

#: Frame version this module writes, and the only one it understands.
VERSION = 1

#: Number of bits the version occupies at the top of the header byte.
VERSION_SHIFT = 4

#: Mask of the message type inside the header byte.
TYPE_MASK = 0x0F

#: Largest version number that fits in the header nibble.
MAX_VERSION = 0x0F

#: Message types a payload may carry.
MSG_PING = 0
MSG_INT = 1
MSG_BLOB = 2
MSG_ECHO = 3

#: Largest payload a single frame may carry.
MAX_PAYLOAD = 4096

#: Largest number of bytes a varint may occupy, for 64 bit values.
MAX_VARINT_BYTES = 10

#: Largest number of bytes a single frame may occupy on the wire.
MAX_FRAME_SIZE = 1 + MAX_VARINT_BYTES + MAX_PAYLOAD + 4

#: Size of the trailing checksum field.
CRC_SIZE = 4

#: Signed range accepted by the zigzag helpers.
SVARINT_MIN = -(2 ** 63)
SVARINT_MAX = 2 ** 63 - 1

#: Highest value a little endian CRC32 field can hold.
_UINT32 = 0xFFFFFFFF


class FrameError(Exception):
    """Raised when a byte stream cannot be decoded as a frame.

    ``incomplete`` is set when the bytes seen so far could still grow into a
    frame: a streaming caller should wait for more bytes instead of reporting
    a broken stream.
    """

    def __init__(self, message, incomplete=False):
        super().__init__(message)
        self.message = message
        self.incomplete = incomplete


class Frame:
    """A decoded frame, holding its version, message type and payload."""

    __slots__ = ("version", "msg_type", "payload")

    def __init__(self, msg_type, payload, version=VERSION):
        self.version = version
        self.msg_type = msg_type
        self.payload = payload

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return NotImplemented
        return (
            self.version == other.version
            and self.msg_type == other.msg_type
            and self.payload == other.payload
        )

    def __hash__(self):
        return hash((self.version, self.msg_type, self.payload))

    def __repr__(self):
        return "Frame(version=%r, msg_type=%r, payload=%r)" % (
            self.version,
            self.msg_type,
            self.payload,
        )


# ---------------------------------------------------------------------------
# Varints
# ---------------------------------------------------------------------------

def encode_uvarint(value):
    """Encode a non negative integer as an unsigned varint."""
    if value < 0:
        raise FrameError("a varint needs a non negative integer")
    chunk = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            chunk.append(byte | 0x80)
            continue
        chunk.append(byte)
        return bytes(chunk)


def decode_uvarint(data, offset=0):
    """Decode the varint at offset, returning (value, bytes consumed)."""
    value = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise FrameError("truncated varint", incomplete=True)
        if pos - offset >= MAX_VARINT_BYTES:
            raise FrameError("the varint is longer than %d bytes" % MAX_VARINT_BYTES)
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos - offset
        shift += 8


def encode_svarint(value):
    """Encode a signed 64 bit integer as a zigzag varint."""
    if not SVARINT_MIN <= value <= SVARINT_MAX:
        raise FrameError("the signed varint is out of range")
    return encode_uvarint((value << 1) ^ (value >> 63))


def decode_svarint(data, offset=0):
    """Decode the zigzag varint at offset, returning (value, bytes consumed)."""
    raw, used = decode_uvarint(data, offset)
    return (raw >> 1) ^ (raw & 1), used


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def _checksum(data):
    """CRC32 of the frame bytes the checksum field covers."""
    return zlib.crc32(data) & _UINT32


def encode_frame(msg_type, payload, version=VERSION):
    """Encode one frame carrying payload."""
    if isinstance(payload, str):
        raise TypeError("payload must be bytes, not text")
    payload = bytes(payload)
    if not 0 <= msg_type <= TYPE_MASK:
        raise FrameError("message type %r does not fit the header" % (msg_type,))
    if not 0 <= version <= MAX_VERSION:
        raise FrameError("version %r does not fit the header" % (version,))
    if len(payload) > MAX_PAYLOAD:
        raise FrameError("payload of %d bytes is above the limit" % len(payload))
    header = (version << VERSION_SHIFT) | msg_type
    body = bytes([header]) + encode_uvarint(len(payload)) + payload
    return body + struct.pack("<I", _checksum(payload))


def decode_frame(data):
    """Decode the frame at the head of data, returning (Frame, bytes consumed)."""
    if not data:
        raise FrameError("truncated frame", incomplete=True)
    header = data[0]
    version = (header >> VERSION_SHIFT) & TYPE_MASK
    msg_type = header & TYPE_MASK
    if version < VERSION:
        raise FrameError("version %d is not supported" % version)
    length, used = decode_uvarint(data, 1)
    if length > MAX_FRAME_SIZE:
        raise FrameError("payload of %d bytes is above the limit" % length)
    body_end = 1 + used + length
    if len(data) < body_end + CRC_SIZE:
        raise FrameError("truncated frame", incomplete=True)
    payload = data[1 + used:body_end]
    stored = struct.unpack_from("<I", data, body_end)[0]
    if stored != _checksum(payload):
        raise FrameError("checksum mismatch")
    return Frame(msg_type, bytes(payload), version=version), body_end + CRC_SIZE


def decode_all(data):
    """Decode every frame held by data.

    A buffer that runs out in the middle of a frame is reported as a
    FrameError: a short list would hide the missing bytes from the caller.
    """
    frames = []
    offset = 0
    while offset < len(data):
        try:
            frame, consumed = decode_frame(data[offset:])
        except FrameError:
            break
        frames.append(frame)
        offset += consumed
    return frames


def encode_int_frame(value, version=VERSION):
    """Encode an integer frame; the payload is a zigzag varint."""
    return encode_frame(MSG_INT, encode_svarint(value), version=version)


def decode_int_frame(frame):
    """Return the integer carried by frame."""
    if frame.msg_type != MSG_INT:
        raise FrameError("frame type %d does not carry an integer" % frame.msg_type)
    value, used = decode_svarint(frame.payload, 0)
    if used != len(frame.payload):
        raise FrameError("the integer frame carries trailing bytes")
    return value


class FrameParser:
    """Incremental reader for frame bytes.

    ``feed`` appends whatever arrived and ``read_frame`` returns the next
    complete frame, or None while the buffer holds only part of one.  Bytes
    that cannot be a frame raise FrameError and stay in the buffer for the
    caller to inspect.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        """Append bytes that arrived from the peer."""
        if isinstance(data, str):
            raise TypeError("feed() takes bytes, not text")
        self._buffer.extend(data)

    def pending(self):
        """Number of buffered bytes that have not been consumed yet."""
        return len(self._buffer)

    def reset(self):
        """Drop every buffered byte."""
        self._buffer = bytearray()

    def read_frame(self):
        """Return the next complete frame, or None while more bytes are needed."""
        if not self._buffer:
            return None
        try:
            frame, consumed = decode_frame(bytes(self._buffer))
        except FrameError:
            return None
        del self._buffer[:consumed]
        return frame
