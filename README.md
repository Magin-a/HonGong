# Learn to AI🚀
-혼자 공부하는 머신러닝+딥러닝\
+) 인공지능 수업 

# 1.Numpy란?
> - numpy는 다차원 배열을 효과적으로처리하는 도구
> - 현실세계의 다양한 데이터를 배열 형태로 표현할 수 있습니다.
> - Python의 기본 list에 비해 빠르고 강력한 기능을 제공함

>  ### Numpy의 차원
>> - 1차원 축(행): axis 0 => Vector
>> - 2차원 축(열): axis 1 => Matrix
> ### Numpy 함수
>> - 배열 초기화
>>  ####- '.arange'\
 = arr1 = np.arange(4) => [0 1 2 3]
>>  ####- 'zeros, ones'  
= arr2 = np.zeros((4, 4), dtype = float)\
 =>  4*4 크기 [[0, 0, 0, 0], [0, 0, 0, 0]]  
 >>  ####- ".random.###" \
 arr3 = np.random.randint(0, 10, (3, 3))

 >> ####- Numpy 배열 변환
 >>  - arr2 = np.arange(8).reshape(2, 4)\
 \-결과- \
[0 1 2 3]         
[4 5 6 7]

>> ####- 배열 나누기
>>  - arr = np.arange(8).reshape(2,4)  \
left, right = np.split(arr, [2], axis =1)   \
\- 결과-  \
left: [[0 1] \,
 [4 5]]  \
right:  [[2 3]\,
 [6 7]]
