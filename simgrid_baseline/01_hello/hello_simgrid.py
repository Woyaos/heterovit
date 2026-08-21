import sys

from simgrid import Actor, Engine, this_actor


def gpu_worker():
    work = 10_000_000
    this_actor.info("Start a task with 10 million simulated operations")
    this_actor.execute(work)
    this_actor.info("Task finished; expected simulated duration: 0.010 s")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python3 {sys.argv[0]} platform.xml")

    engine = Engine(sys.argv)
    engine.load_platform(sys.argv[1])
    Actor.create("gpu-worker", engine.host_by_name("GPU"), gpu_worker)
    engine.run()
    print(f"RESULT simulated_time_s={engine.clock:.6f}")


if __name__ == "__main__":
    main()
