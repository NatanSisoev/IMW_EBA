from src.simulator import BusSimulator
from src.visualizer import visualize_bus_stations

if __name__ == "__main__":
    print(" Bus Simulator ".center(80, "="))

    print("Initializing simulator...", end=" ")
    data_filename = "data/data.csv"
    bs = BusSimulator.read_csv(data_filename)
    print("DONE!")

    print("Example workflow:")
    print("\t0. visualize_bus_stations() (e.g. visualize_bus_stations(\"data/data.csv\")")
    print("\t1. bs.run(HH) (e.g. bs.run(17, traffic=False, traffic_lights=False))")
    print("\t2. bs.trip_time (in seconds)")
    print("\t3. bs.plot() (interactive plot)")

    while True:
        try:
            line = input("> ")
            if line.strip().lower() in {"exit", "quit"}:
                break
            try:
                result = eval(line)
                if result is not None:
                    print(result)
            except SyntaxError:
                exec(line)
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
