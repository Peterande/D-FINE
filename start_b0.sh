CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=23556 --nproc_per_node=4 tools/train.py -c /home/pengys/code/rtdetrv2_pytorch/configs/dfine/objects365/dfine_hgnetv2_s_1x_obj365.yml --use-amp --seed=0 --output-dir tb0902/b0_obj365_fix -r /home/pengys/code/rtdetrv2_pytorch/tb0902/b0_obj365_fix/17.pth
if [ $? -ne 0 ]; then
    echo "First training failed, restarting with resume option..."
    while true; do
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=23557 --nproc_per_node=4 tools/train.py -c /home/pengys/code/rtdetrv2_pytorch/configs/dfine/objects365/dfine_hgnetv2_s_1x_obj365.yml --use-amp --seed=0 --output-dir tb0902/b0_obj365_fix -r tb0902/b0_obj365_fix/last.pth
        if [ $? -eq 0 ]; then
            break
        fi
    done
fi
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=25554 --nproc_per_node=4 tools/train.py -c /home/pengys/code/rtdetrv2_pytorch/configs/dfine/objects365/dfine_hgnetv2_m_1x_obj365.yml --use-amp --seed=0 --output-dir tb0902/b2_obj365_fix -r tb0902/b2_obj365_fix/last.pth
# if [ $? -ne 0 ]; then
#     echo "First training failed, restarting with resume option..."
#     while true; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=25554 --nproc_per_node=4 tools/train.py -c /home/pengys/code/rtdetrv2_pytorch/configs/dfine/objects365/dfine_hgnetv2_m_1x_obj365.yml --use-amp --seed=0 --output-dir tb0902/b2_obj365_fix -r tb0902/b2_obj365_fix/last.pth
#         if [ $? -eq 0 ]; then
#             break
#         fi
#     done
# fi