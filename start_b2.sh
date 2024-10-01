CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=23556 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/dfine_hgnetv2_m_1x_obj365.yml --use-amp --seed=0 --output-dir tb/B2 -r tb/B2/12.pth
if [ $? -ne 0 ]; then
    echo "First training failed, restarting with resume option..."
    while true; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=23557 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/dfine_hgnetv2_m_1x_obj365.yml --use-amp --seed=0 --output-dir tb/B2 -r tb/B2/last.pth
        if [ $? -eq 0 ]; then
            break
        fi
    done
fi