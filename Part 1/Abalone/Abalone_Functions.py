import numpy as np
import csv
import time

np.random.seed(1234)

RND_MEAN = 0  # 난수 평균
RND_STD = 0.0030  # 난수 표준편차

LEARNING_RATE = 0.001


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


def load_abalone_dataset():
    """
    csv 파일을 메모리로 불러오고, 둘 째 줄부터의 데이터를 리스트로 수집한다.
    데이터 리스트와 입력 데이터 개수, 출력 데이터 개수를 전역변수로 선언한다.
    비선형적인 데이터를 원-핫 벡터로 변경하고 나머지 데이터또한 데이터리스트에 담는다.
    """
    with open('Abalone_Data.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # 첫 줄 건너뛰기
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + output_cnt])

    for n, row in enumerate(rows):
        if row[0] == 'I':
            data[n, 0] = 1
        if row[0] == 'M':
            data[n, 1] = 1
        if row[0] == 'F':
            data[n, 2] = 1
        data[n, 3:] = row[1:]


def init_model():
    """
    모델의 가중치 / 편향 텐서를 생성
    정규분포를 따르는 난수로 초기화, bias 의 경우 초기의 큰 영향을 고려해 0으로 초기화
    """
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])  # 정규분포 난수 생성
    bias = np.zeros([output_cnt])


def train_and_test(epoch_count, mb_size, report):
    """
    정해진 에포크와 미니배치의 수를 이용해 반복문에서 학습진행
    정해진 주기마다 시험을 통해 정확도 및 누적 정확도/비용 출력

    :param epoch_count: 학습 총 에포크 수
    :param mb_size: 미니배치 크기
    :param report: 중간보고 주기
    """
    step_count = arrange_data(mb_size)  # mini batch 의 수
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []  # 비용, 정확도

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print("Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}".
                  format(epoch + 1, np.mean(losses), np.mean(accs), acc))

    final_acc = run_test(test_x, test_y)
    print("\nFinal Test: final accuracy = {:5.3f}".format(final_acc))


def arrange_data(mb_size):
    """
    학습 최초에만 한 번 호출
    데이터들의 셔플 맵을 섞고, 미니배치의 크기를 토대로 0.8 학습 데이터와 0.2 평가데이터의 구분점을 생성

    :param mb_size: 미니 배치의 크기
    :return: 미니 배치의 수
    """
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)

    step_count = int(data.shape[0] * 0.8) // mb_size  # mini batch 의 수. 80%를 학습데이터로, 나머지를 평가데이터로 사용
    test_begin_idx = step_count * mb_size
    return step_count


def get_test_data():
    """
    data 리스트에서 셔플맵을 토대로 평가데이터를 추출

    :return: 평가데이터 텐서(입력데이터, 출력데이터)
    """
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


def get_train_data(mb_size, n):
    """
    data 리스트에서 셔플맵을 토대로 학습데이터를 추출
    각 에포크의 첫번째 마다 학습데이터들의 셔플맵을 섞어 에포크마다 다른순서로 학습이 되게 함

    :param mb_size: 미니배치 크기
    :param n: 현재 미니배치의 인덱스
    :return: 학습데이터 텐서(입력데이터, 출력데이터)
    """
    global data, shuffle_map, test_begin_idx, output_cnt
    if n == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * n: mb_size * (n + 1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]


def run_train(x, y):
    """
    데이터들을 통해 순전파, 후처리 순으로 비용/정확도 계산
    순전파/후처리의 결과를 토대로 손실기울기 계산, 역전파 수행

    :param x: 학습 입력 데이터
    :param y: 학습 출력 데이터
    :return: 이번 배치 순전파 결과의 비용,정확도
    """
    output, aux_nn = forward_neuralent(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0  # 손실함수에 대한 손실함수의 기울기(1) ∂L/∂L
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)  # 역전파를 통한 학습

    return loss, accuracy


def run_test(x, y):
    """
    평가 데이터를 통해 정확도 계산

    :param x: 평가 입력 데이터
    :param y: 평가 출력 데이터
    :return: 출력 정확도
    """
    output, _ = forward_neuralent(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def forward_neuralent(x):
    """
    순전파 계산

    :param x: 입력 데이터
    :return: 신경망 출력 값, 신경망 입력 값
    """
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x


def forward_postproc(output, y):
    """
    출력과 데이터를 통해 비용, 차이 계산

    :param output: 신경망 출력 값
    :param y: 출력 데이터
    :return: 비용, 오차
    """
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_neuralnet(G_output, x):
    """
    역전파를 수행하여 학습

    :param G_output: 신경망 출력
    :param x: 신경망 입력데이터
    """
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b


def backprop_postproc(G_loss, diff):
    """
    출력결과와 출력데이터의 오차를 통해 출력에 대한 손실함수의 기울기 ∂L/∂y 를 구한다

    :param G_loss: 손실함수에 대한 손실함수의 기울기(1)
    :param diff: 오차
    :return: 출력에 대한 손실함수의 기울기
    """
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output


def eval_accuracy(output, y):
    """
    출력 결과와 출력 데이터를 통해 정확도 계산

    :param output: 신경망 출력
    :param y: 출력 데이터
    :return: 정확도(0 ~ 1)
    """
    mdiff = np.mean(np.abs((output - y) / y))
    return 1 - mdiff
