if [ ! -f "images_background.zip" ]; then
    curl -O https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip
fi
if [ ! -f "images_evaluation.zip" ]; then
    curl -O https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip
fi

mkdir -p omniglot_raw_data

if [ ! -d "omniglot_raw_data/images_background" ]; then
    unzip images_background.zip -d omniglot_raw_data
fi
if [ ! -d "omniglot_raw_data/images_evaluation" ]; then
    unzip images_evaluation.zip -d omniglot_raw_data
fi

rm -rf images_background.zip
rm -rf images_evaluation.zip
