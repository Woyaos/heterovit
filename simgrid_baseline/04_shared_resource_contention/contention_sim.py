import sys

from simgrid import Actor, Engine, Mailbox, this_actor


TENSOR_BYTES = 4_000_000


def gpu_request(request_id):
    to_fpga = Mailbox.by_name(f"request-{request_id}-to-fpga")
    to_gpu = Mailbox.by_name(f"request-{request_id}-to-gpu")

    this_actor.info(f"Request {request_id}: GPU stage A starts")
    this_actor.execute(1_000_000)
    this_actor.info(f"Request {request_id}: sends 4 MB to FPGA")
    to_fpga.put(f"input-{request_id}", TENSOR_BYTES)
    to_gpu.get()
    this_actor.info(f"Request {request_id}: GPU stage C starts")
    this_actor.execute(1_000_000)
    this_actor.info(f"Request {request_id}: finished")


def fpga_request(request_id):
    to_fpga = Mailbox.by_name(f"request-{request_id}-to-fpga")
    to_gpu = Mailbox.by_name(f"request-{request_id}-to-gpu")

    to_fpga.get()
    this_actor.info(f"Request {request_id}: FPGA Linear starts")
    this_actor.execute(400_000)
    this_actor.info(f"Request {request_id}: returns 4 MB to GPU")
    to_gpu.put(f"output-{request_id}", TENSOR_BYTES)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: python3 {sys.argv[0]} <request-count> <platform.xml>")

    request_count = int(sys.argv[1])
    if request_count < 1:
        raise SystemExit("request-count must be positive")

    engine = Engine([sys.argv[0]])
    Engine.set_config("network/model:CM02")
    engine.load_platform(sys.argv[2])
    gpu = engine.host_by_name("GPU")
    fpga = engine.host_by_name("FPGA")

    for request_id in range(1, request_count + 1):
        Actor.create(f"gpu-request-{request_id}", gpu, gpu_request, request_id)
        Actor.create(f"fpga-request-{request_id}", fpga, fpga_request, request_id)

    engine.run()
    print(f"RESULT requests={request_count} makespan_ms={engine.clock * 1000:.6f}")


if __name__ == "__main__":
    main()
