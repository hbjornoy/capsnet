cd language_dataset
echo `pwd`
ROOT=`pwd`

for traintest in */ ; do
    echo "$traintest"
    for language in */ ; do
      echo "$language"
      TEST= `find . -mindepth 1 -type f -print0 | sed 's/$/ .     /'`
      echo "$TEST"
      #find . -mindepth 1 -type f -print0 | sed 's/$/ ./' | xargs -0 -L1 mv { .}
    done
done
#find . -mindepth 1 -type f -print0 | sed 's/$/ ./' | xargs -0 -L10 mv
#find ./ -mindepth 2 -type f -print0 | sed 's/$/ ./' | xargs -0 -L10 mv


