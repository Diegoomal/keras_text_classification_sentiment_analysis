echo "===== bash destroy ====="

echo "1) Remove conda env"

conda deactivate

conda remove --name project-env --all -y

echo "2) Remove dir and files"

rm -rf __pycache__
rm -rf .pytest_cache

rm -rf segment-anything

rm -rf src/__pycache__
rm -rf src/images/dog.jpg
rm -rf src/images
rm -rf src/sam_vit_h_4b8939.pth

rm -rf docs

rm -rf tests/__pycache__