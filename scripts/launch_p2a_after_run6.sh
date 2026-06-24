#!/usr/bin/env bash
# P2a（D32）自动接力：等 train_run6 进程结束 → 启动 NAFNet 同预算对照 run（train_run7）。
# 用法：在后台跑本脚本；它会阻塞到 train_run6 退出后再 setsid 拉起 P2a。
set -u
PY=/home/jing/anaconda3/envs/bokeh/bin/python
ROOT=/home/jing/bokeh
RUN6_PID=151612                                   # train_run6 工作进程

cd "$ROOT" || exit 1

# 1) 等 train_run6 结束（进程消失）。
while kill -0 "$RUN6_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] train_run6 (pid $RUN6_PID) 已结束，$(date)" >> outputs/p2a_chain.log

# 2) 留 20s 让显存释放干净，再启动 P2a。
sleep 20

# 3) 启动 P2a：NAFNet 块、同预算(144/72)、50k、AMP、batch6、COCO 背景。
mkdir -p outputs/train_run7
setsid "$PY" -u -m train.train --iters 50000 --amp --batch 6 \
    --block naf --ar-mid 144 --iu-mid 72 \
    --bg-dirs /home/jing/datasets/coco/images/train2017 \
    --out outputs/train_run7 \
    > outputs/train_run7_launch.log 2>&1 < /dev/null &
echo "[chain] P2a (train_run7) 已启动 pid $!，$(date)" >> outputs/p2a_chain.log
