import sys

from simgrid import Actor, Engine, Mailbox, this_actor


TENSOR_BYTES = 4_000_000


def gpu_worker():
    this_actor.info("Stage A: GPU compute starts")
    this_actor.execute(1_000_000)  # 1 ms on the virtual 1 Gf/s GPU

    this_actor.info("Transfer input: GPU -> FPGA")
    Mailbox.by_name("gpu-to-fpga").put("input tensor", TENSOR_BYTES)

    this_actor.info("GPU waits for the FPGA result")
    result = Mailbox.by_name("fpga-to-gpu").get()
    this_actor.info(f"Received {result}; Stage C: GPU compute starts")
    this_actor.execute(1_000_000)  # another 1 ms
    this_actor.info("Pipeline finished")


def fpga_worker():
    tensor = Mailbox.by_name("gpu-to-fpga").get()
    this_actor.info(f"Received {tensor}; Stage B: FPGA compute starts")
    this_actor.execute(400_000)  # 0.4 ms on the virtual 1 Gf/s FPGA

    this_actor.info("Transfer output: FPGA -> GPU")
    Mailbox.by_name("fpga-to-gpu").put("output tensor", TENSOR_BYTES)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python3 {sys.argv[0]} platform.xml")

    engine = Engine(sys.argv)
    # CM02 keeps this first experiment close to latency + bytes / bandwidth.
    Engine.set_config("network/model:CM02")
    engine.load_platform(sys.argv[1])
    Actor.create("gpu-worker", engine.host_by_name("GPU"), gpu_worker)
    Actor.create("fpga-worker", engine.host_by_name("FPGA"), fpga_worker)
    engine.run()
    print(f"RESULT pipeline_time_s={engine.clock:.6f}")


if __name__ == "__main__":
    main()
