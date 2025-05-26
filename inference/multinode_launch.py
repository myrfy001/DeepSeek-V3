from fabric import Connection, Config, ThreadingGroup
from threading import Thread

hosts = [(f"10.18.17.{i+141}", f"h02r3n{i:02d}") for i in range(3, 3+15)]
password = "XaNjj@##Apir!"


def add_docker_to_user_group(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    c.sudo("usermod -aG docker $USER")
    c.close()

def stop_all(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    r = c.run(f"""docker stop ds_pp || true && \
              docker rm ds_pp || true && \
              docker ps | grep ds_pp
              """)
    c.close()

def run_model(node_id, ip_addr, user_name):
    config = Config(overrides={'sudo': {'password': password}})
    c = Connection(ip_addr, user=user_name, connect_kwargs={"password": password}, config=config)
    r = c.run(f"""
              export IMAGE=image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.3.0-ubuntu22.04-dtk24.04.3-py3.10 && \
              docker run --rm \
              --name ds_pp \
              --network=host \
              --ipc=host \
              --shm-size=16G \
              --device=/dev/kfd \
              --device=/dev/mkfd \
              --device=/dev/dri \
              --device=/dev/infiniband/uverbs0 \
              --device=/dev/infiniband/uverbs3 \
              --device=/dev/infiniband/rdma_cm \
              --cap-add=IPC_LOCK \
              -v /htxjj:/data \
              -v /opt/hyhal:/opt/hyhal \
              --group-add video \
              --cap-add=SYS_PTRACE \
              --security-opt seccomp=unconfined \
              $IMAGE \
              /bin/bash /data/mmh/launch_model.sh {node_id}
              """)
    c.close()


threads = []
for node_id, (ip_addr, user_name) in enumerate(hosts):
    node_id = node_id + 1
    threads.append(Thread(target=stop_all, args=(node_id, ip_addr, user_name)))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()


threads = []
for node_id, (ip_addr, user_name) in enumerate(hosts):
    node_id = node_id + 1
    threads.append(Thread(target=run_model, args=(node_id, ip_addr, user_name)))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()