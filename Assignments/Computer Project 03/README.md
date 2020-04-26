# CSE 6363 Machine Learning

- Student ID: 1001778270
- Name: Bo Lin

## Computer Project 3

- Python: 3.7.7
- Numpy: 1.18.2
- Scikit-Learn: 0.22.2.post1

**Computer Project 3A:** in `main.py`

**Computer Project 3B:** in `main.py`

## Usage

Execute the following command in your terminal

```sh
python main.py
```

## Results

- AT&T Face Images

39965.30311 Acc: 0.79750
|      |             |
| ---- | ----------- |
| Loss | 39965.30311 |
| Acc  | 0.79750     |

```plain
confusion matrix:
6  1  0  0  0  0  0  0  0  0  0  0  2  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  9  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  1  0  9  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  9  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  1  0  0  9  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
2  0  0  0  0  0  0  0  0  0  0  0  0  0  2  6  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  3  0  0  0  0  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  9  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  7  3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0  0  0  1  0  0  0
0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  3  0  0  0  0  0  0  0  0  0  0  6  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  5  0  0  0  0  0  0  0  0  4  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  4  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0  0  0  0
0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  9  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  8  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0  0  0  0
0  0  0  3  3  0  0  0  0  0  0  0  0  0  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  3  0  0  0  0  0  0  0  1  0  6  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  0  0  0  0  0  0  0  0  5  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0
0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```

- Hand Written Letters

|      |           |
| ---- | --------- |
| Loss | 411.71804 |
| Acc  | 0.52761   |

```plain
confusion matrix:
17 1  0  0  0  0  0  7  0  0  1  0  2  0  0  4  4  0  0  0  0  0  3  0  0  0
1 19  0  1  1  0  0  0  0  0  0  4  0  0  2  2  0  2  7  0  0  0  0  0  0  0
0  0 20  0  0  0  0  0  0  0  0  9  0  0 10  0  0  0  0  0  0  0  0  0  0  0
0 10  1  7  0  0  0  0  2  0  0  1  0  0 10  0  2  0  1  0  0  0  0  0  0  5
0  0  1  0 31  0  0  0  0  0  0  4  0  0  1  1  0  0  0  0  0  0  0  0  0  1
2  0  0  0  0 19  0  0  1  0  0  1  0  0  0 16  0  0  0  0  0  0  0  0  0  0
0  0  8  3  0  1  0  0  0  0  0  3  0  0  1  0 19  0  0  0  1  0  0  0  3  0
0  0  0  0  0  0  0 17  0  0  0  1  9  0  0  3  0  0  0  0  2  0  4  1  2  0
0  0  0  0  0  1  0  0 24  9  0  0  0  0  0  0  1  0  0  0  0  0  0  1  2  1
0  0  0  0  0  0  0  0  2 24  0  2  0  0  0  0  1  0  9  0  0  0  0  0  1  0
0  0  0  0  0  0  0  0  0  0 19  5  0  0  0  4  0  0  0  0  0  0  0  8  3  0
0  0  0  0  0  0  0  0  3  0  0 34  0  0  0  0  0  0  0  0  0  0  0  0  0  2
1  1  0  1  0  0  0  7  0  0  1  0 25  0  0  1  0  0  0  0  1  0  1  0  0  0
0  0  0  2  0  0  0  4  0  0  2  1 10  0  0  4  0  0  0  0  1  0 13  0  2  0
0  2  5  5  0  0  0  0  0  0  0  3  0  0 24  0  0  0  0  0  0  0  0  0  0  0
0  0  0  6  0  2  0  0  0  0  0  0  0  0  0 29  1  1  0  0  0  0  0  0  0  0
1  0  2  4  0  0  0  1  0  0  0  1  0  0  1  1 28  0  0  0  0  0  0  0  0  0
1  0  1  0  1  1  0  0  0  0  0  3  0  0  0  9  1 20  0  0  0  0  0  1  0  1
0  5  2  0  1  0  0  0  0  2  0  2  0  0  0  0  1  0 26  0  0  0  0  0  0  0
0  0  0  0  0  3  0  0 17 12  0  0  0  0  0  4  1  0  0  0  0  0  0  0  2  0
0  0  0  6  0  0  0  0  0  0  1  3  0  0  0  0  1  1  0  0 25  2  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3 21  2  0 13  0
0  0  0  0  0  0  0  0  0  0  1  2  0  0  0  0  0  0  0  0  4  4 24  3  1  0
0  0  0  0  0  1  0  0  0  0 13  0  0  0  0  0  0  0  0  0  0  0  0 18  7  0
0  0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  0  0  0  0  0  2  3  0 32  0
0  0  0  0  0  0  0  0  2  2  0  3  0  0  0  0  0  0  0  0  0  0  0  0  0 32
```