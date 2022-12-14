set -o errexit
cd src
git clone https://github.com/huggingface/transformers.git
cd transformers
git config core.filemode false
git checkout v4.11.3
# apply patch
git apply ../../scripts/transformers-v4.11.3-x.patch
pip install .
