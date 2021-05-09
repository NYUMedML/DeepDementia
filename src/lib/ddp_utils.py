import os
import signal
import threading

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn as nn


EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
signal.signal(signal.SIGUSR2, _clean_exit_handler)


def get_ifname():
    return ifcfg.default_interface()["device"]


def init_distrib_lsf(backend="nccl"):
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8738'

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("LSB_DJOB_NUMPROC", 1)))

    print('Local rank: {:d} World rank: {:d} Wold size {:d}'.format(local_rank, world_rank, world_size))

    #tcp_store = distrib.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    distrib.init_process_group(
        backend, rank=world_rank, world_size=world_size
    )

    return local_rank


def convert_groupnorm_model(module, ngroups=32):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))

    return mod
    