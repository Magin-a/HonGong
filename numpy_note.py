#numpy 불러오기
import numpy as np
data1 = [1, 2, 3, 4, 5]
print(data1)
print(type(data1))

#배열 형태 변환
arr1 = np.array(data1)
print(arr1, type(arr1))

#배열의 크기 확인하기
print(arr1.shape())

#파이썬 range 함수와 같지만 ndarray객체를 생성 .arange(start, end, 간격)
arr2 = np.arange(1, 7, 2) # 출력: [1, 3, 5]

#np.zeros() 함수는 인자로 받은 값의 크기만큼 모든 값이 0인 array 생성
np.zeros() #5입력 => 출력: array([0, 0, 0, 0, 0]) 

#numpy의 주요 데이터형 dtype( int, float 등 출력)
arr2.dtype()

#2차원 리스트 형태 입력 배열 생성
arr4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#연산(사칙연산)
arr1 = [[1, 2, 3] ,[4, 5, 6]]
arr2 = [[7, 8, 9], [10, 11, 12]]

a,b,c,d = np.add(arr1, arr2), np.substract(arr1, arr2), np.multiply(arr1, arr2), np.divide(arr2, arr1)
a,b,c,d = arr1+arr2, arr1-arr2, arr1*arr2, arr2/arr1

#array 브로드캐스트 
#서로 다른 크기의 t1, t2 연산 
t1 = np.array([1, 2, 3])
t2 = np.array([[4, 5, 6], [7, 8, 9]])
# '+' 출력 => [[5, 7, 9], [8, 10, 12]]


#array boolean 인덱싱(마스크)
names = np.array(['kim', 'lee', 'park', 'moon'])
names_mask = (names == 'kim') #조건에 해당하는 마스크
print(names[names_mask])
#출력: [True False False False]
#출력: [0 0] 배열의 정보

print(names[names_mask,:]) #arr1[행,열]  , ':' 모든 데이터를 뜻함
#출력: mask조건에 맞는 행을 전체 출력

#예시 0번째 열의 값이 0보다 작은 행의 2, 3번째 열 값
data = np.random.randn(5 ,4) 
data[data[:,0]<0, 2:4] = 0

#다른 연산 함수(.sqrt(arr1), .square(arr1), .exp(arr1), .log(arr1), .log10(), .sign())
arr1 = np.array([1 ,2 ,3, 4])

#절대값, 각 성분의 제곱근(== arr1**0.5)
np.abs(arr1) , np.sqrt(arr1)

#각 성분의 제곱, 무리수 e의 지수로 값을 계산, log, log10 등등
np.square(arr1), np.exp(arr1), np.log10(arr1)

#각 성분의 부호성분 계산, 각 성분의 소수 첫 번째 자리에서 올림 값, 내림
np.sign(arr1), np.ceil(arr1), np.floor(arr1)

#각 성분이 nan인 경우 True를 아닌 경우 False를 변환하기, 각 성분이 무한대인 경우 True를 그외 False
np.isnan(arr1), np.isinf(arr1) 

#삼각함수(cos(), sin(), tan()) 'h'붙은건 쌍곡선 함수 
np.sin(arr1), np.tan(arr1), np.sinh(arr1)

#다차원 배열(axis)
#axis=0 x축만 값을 가진 Row, axis = 1 x,y의 값을 가지는 행렬(depth==1), 
# axist=2 속칭(tensor) 행과 열을 갖고 각 컬럼은 벡터 형태 Depth 
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
np.sum(arr1) #모든 성분 값 구분 없이 더한 값
np.sum(arr1, axis=0) #같은 row위치 값 출력 : [5, 7, 9]
np.sum(arr1, axis=1) #같은 행위치 값 출력 : [6, 15]

#그외 함수(.mean(평균 값), .std(표준편차), .min(최소), .var(분산))
np.mean(arr1), np.std(arr1), np.min(arr1), np.var(arr1)