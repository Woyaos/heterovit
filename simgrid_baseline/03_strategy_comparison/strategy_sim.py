import sys

from simgrid import Actor, Engine, Mailbox, this_actor


TENSOR_BYTES = 4_000_000


def gpu_only_worker():
    this_actor.info("A on GPU: 1.0 ms")
    this_actor.execute(1_000_000)
    this_actor.info("Linear on GPU: 2.0 ms")
    this_actor.execute(2_000_000)
    this_actor.info("C on GPU: 1.0 ms")
    this_actor.execute(1_000_000)


def offload_gpu_worker():
    this_actor.info("A on GPU: 1.0 ms")
    this_actor.execute(1_000_000)
    this_actor.info("Send 4 MB Linear input to FPGA")
    Mailbox.by_name("gpu-to-fpga").put("linear input", TENSOR_BYTES)
    Mailbox.by_name("fpga-to-gpu").get()
    this_actor.info("C on GPU: 1.0 ms")
    this_actor.execute(1_000_000)


def offload_fpga_worker():
    Mailbox.by_name("gpu-to-fpga").get()
    this_actor.info("Linear on FPGA: 0.4 ms")
    this_actor.execute(400_000)
    this_actor.info("Return 4 MB Linear output to GPU")
    Mailbox.by_name("fpga-to-gpu").put("linear output", TENSOR_BYTES)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            f"Usage: python3 {sys.argv[0]} <gpu-only|offload> <platform.xml>"
        )

    strategy = sys.argv[1]
    platform = sys.argv[2]
    if strategy not in {"gpu-only", "offload"}:
        raise SystemExit(f"Unknown strategy: {strategy}")

    engine = Engine([sys.argv[0]])
    Engine.set_config("network/model:CM02")
    engine.load_platform(platform)

    gpu = engine.host_by_name("GPU")
    if strategy == "gpu-only":
        Actor.create("gpu-only", gpu, gpu_only_worker)
    else:
        fpga = engine.host_by_name("FPGA")
        Actor.create("offload-gpu", gpu, offload_gpu_worker)
        Actor.create("offload-fpga", fpga, offload_fpga_worker)

    engine.run()
    print(f"RESULT strategy={strategy} time_ms={engine.clock * 1000:.6f}")


if __name__ == "__main__":
    main()
