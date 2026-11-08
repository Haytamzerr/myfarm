"""Command line:

    python -m myfarm serve [--port 8000]                    the map dashboard
    python -m myfarm scan [--source demo|sentinel] [--start ..] [--end ..] [--out zones.csv]
"""

import argparse
import json

from myfarm.api import PARCEL_FILE
from myfarm.indices import add_indices
from myfarm.parcel import load_kml
from myfarm.zones import best_k, management_zones


def main() -> None:
    parser = argparse.ArgumentParser(prog="myfarm")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the map dashboard")
    serve.add_argument("--port", type=int, default=8000)
    scan = sub.add_parser("scan", help="compute indices and management zones")
    scan.add_argument("--source", choices=["demo", "sentinel"], default="demo")
    scan.add_argument("--start", default="2025-06-01")
    scan.add_argument("--end", default="2025-07-15")
    scan.add_argument("--k", type=int, default=0, help="number of zones (0 = choose the best)")
    scan.add_argument("--out", help="write every pixel to this CSV file")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        print(f"\n  MyFarm is running: open http://localhost:{args.port}\n")
        uvicorn.run("myfarm.api:app", host="127.0.0.1", port=args.port)
        return

    parcel = load_kml(PARCEL_FILE)
    points = parcel.grid(10.0)
    if args.source == "demo":
        from myfarm.synthetic import simulate_scene

        bands, meta = simulate_scene(parcel, points), {"source": "demo (simulated)"}
    else:
        from myfarm import sentinel

        bands, meta = sentinel.fetch(parcel, points, args.start, args.end)
    table = add_indices(bands)
    k = args.k or best_k(table)
    zoned, summary = management_zones(table, k)
    print(f"Parcel {parcel.name}: {parcel.area_ha:.2f} ha, {len(points)} pixels of 10 m")
    print(json.dumps({"meta": meta, **summary}, indent=2, default=str))
    if args.out:
        zoned.to_csv(args.out, index=False)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
