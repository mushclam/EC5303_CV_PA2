# Computer Vision Programming Assignment 2

본 자료는 광주과학기술원 AI대학원 `EC5303 Computer Vision`의 `Programming Assignment 2 - Structure from Motion`에 대한 코드 및 보고서를 포함한다.

본 자료에 대한 보고서는 `report.ipynb`에 서술되어 있으며, 문제의 각 문항에 대한 코드가 실행 가능하도록 구현되어 있다.

코드의 구성은 다음과 같다.
1. `main.py` 전체 코드 실행 조작 제공
2. `simpleMatcher.py` feature matching을 위한 matcher 조작 클래스
3. `ransac.py` RANSAC을 이용한 essential matrix estimation 함수
4. `utils.py` feature extraction, matrix decomposition, triangulation 및 편의 함수 포함

## Installation

### Install octave
```
# apt-get install octave
```

### Install packages
```
$ pip install -r requirements.txt
```

## Usage
```
usage: main.py [-h] [--intrinsic INTRINSIC] [--input_file1 INPUT_FILE1] [--input_file2 INPUT_FILE2] [--matcher {normal,knn,flann}] [--match_ratio MATCH_RATIO] [--threshold THRESHOLD]
               [--confidence CONFIDENCE] [--output_file OUTPUT_FILE]

optional arguments:
  -h, --help                            show this help message and exit
  --intrinsic INTRINSIC, -K INTRINSIC   File path of intrinsic paramter
  --input_file1 INPUT_FILE1             Path to reference image file
  --input_file2 INPUT_FILE2             Path to input image file
  --matcher {normal,knn,flann}
  --match_ratio MATCH_RATIO             Ratio to select matching points
  --threshold THRESHOLD                 Threshold for RANSAC
  --confidence CONFIDENCE               Confidence for RANSAC
  --output_file OUTPUT_FILE             Path to output 3D point file
```
### Run file
```
python main.py
```