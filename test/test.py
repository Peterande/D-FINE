import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:
    print(
        "Missing dependency: pyserial\n"
        "Install it in your active environment with:\n"
        "  python3 -m pip install pyserial",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read raw UART data from a serial port and print it."
    )
    parser.add_argument(
        "port",
        nargs="?",
        default="/dev/ttyUSB0",
        help="Serial port, for example /dev/ttyUSB0 or COM3",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=38400,
        help="Serial baudrate (default: 38400)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Read timeout in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected serial ports and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports detected.")
            return 1

        print("Detected serial ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
        return 0

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
    except serial.SerialException as exc:
        print(f"Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"Listening on {args.port} at {args.baudrate} baud. Press Ctrl+C to stop.")

    try:
        while True:
            waiting = ser.in_waiting
            chunk = ser.read(waiting or 1)

            if not chunk:
                continue

            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Raw bytes: {chunk!r}")
            print(f"[{timestamp}] Hex      : {chunk.hex(' ')}")

            try:
                decoded = chunk.decode("utf-8")
                print(f"[{timestamp}] Text     : {decoded.rstrip()}")
            except UnicodeDecodeError:
                decoded = chunk.decode("latin-1", errors="replace")
                print(f"[{timestamp}] Text*    : {decoded.rstrip()}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
