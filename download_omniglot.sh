if [ ! -f "images_background.zip" ]; then
    curl -O http://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
fi
if [ ! -f "images_evaluation.zip" ]; then
    curl -O http://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
fi

mkdir -p language_dataset

if [ ! -d "language_dataset/images_background" ]; then
    unzip images_background.zip -d language_dataset
fi
if [ ! -d "language_dataset/images_evaluation" ]; then
    unzip images_evaluation.zip -d language_dataset
fi

rm -rf images_background.zip
rm -rf images_evaluation.zip
