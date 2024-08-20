# tar the folder and submit shell
# usage: bash submit.sh [new_model_path]

if [ ! -n "$1" ]; then
  echo "model path is NULL"
  exit 0
fi

timescope=`date +"%d%H%M"`

echo "copy $1 to $timescope/best.pth and prepare to tar version $timescope"
mkdir $timescope
cp -r ultralytics $timescope/ultralytics
cp deploy/customize_service.py $timescope/customize_service.py
cp deploy/config.json $timescope/config.json
cp $1 $timescope/best.pt

obsutil cp $timescope obs://pcb-linj0866/yolov10/ -r -f
echo "文件已上传至  obs://pcb-linj0866/yolov10/$timescope"

# sleep 3s

tar -cvzf deploy/pcb_yolov10_$timescope.tar.gz $timescope
rm -rf $timescope
