wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/2
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O FD-A.zip https://figshare.com/ndownloader/articles/19930205/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O FD-B.zip https://figshare.com/ndownloader/articles/19930226/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O HAR.zip https://figshare.com/ndownloader/articles/19930244/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O Gesture.zip https://figshare.com/ndownloader/articles/19930247/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O ECG.zip https://figshare.com/ndownloader/articles/19930253/versions/1
wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" -O EMG.zip https://figshare.com/ndownloader/articles/19930250/versions/1

unzip SleepEEG.zip -d src/ts_tcc/data/SleepEEG/
unzip  Epilepsy.zip -d src/ts_tcc/data/Epilepsy/
unzip  FD-A.zip -d src/ts_tcc/data/FD_A/
unzip  FD-B.zip -d src/ts_tcc/data/FD_B/
unzip  HAR.zip -d src/ts_tcc/data/HAR/
unzip  Gesture.zip -d src/ts_tcc/data/Gesture/
unzip  ECG.zip -d src/ts_tcc/data/ECG/
unzip  EMG.zip -d src/ts_tcc/data/EMG/

rm {SleepEEG,Epilepsy,FD-A,FD-B,HAR,Gesture,ECG,EMG}.zip