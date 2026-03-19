wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/2
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O FD-B.zip https://figshare.com/ndownloader/articles/19930226/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O Gesture.zip https://figshare.com/ndownloader/articles/19930247/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O EMG.zip https://figshare.com/ndownloader/articles/19930250/versions/1

unzip SleepEEG.zip -d data/SleepEEG/
unzip  Epilepsy.zip -d data/Epilepsy/
unzip  FD-B.zip -d data/FD/
unzip  Gesture.zip -d data/Gesture/
unzip  EMG.zip -d data/EMG/

rm {SleepEEG,Epilepsy,FD-B,Gesture,EMG}.zip

# Convert datasets to NormWear format
python data_converters/convert_emg_data.py
python data_converters/convert_epilepsy_data.py 
python data_converters/convert_fd_data.py
python data_converters/convert_gesture_data.py
python data_converters/convert_sleepeeg_data.py
