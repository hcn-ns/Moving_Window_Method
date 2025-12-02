import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------#
#各種サンプルパス生成用の関数，データセットの準備#
#----------------------------------------#

# 標準正規分布に従うiidのシミュレーション
def std_normal_iid(N):
    return np.random.standard_normal(N)

# 株価のサンプルパスのシミュレーション
# 簡単のため株価は金利:r, ボラティリティ:sigmaの幾何ブラウン運動に従うとする
def stock_price(S0, T, r, sigma, M):
    dt = T/M
    random_points = np.random.standard_normal(M)
    log_dSt = (r -sigma **2 /2) *dt + sigma * np.sqrt(dt) * random_points
    log_ST = np.append(0, np.cumsum(log_dSt))
    ST = np.exp(log_ST) * S0
    return ST

# 日次データseqが与えられた時，対数収益率のサンプルパスを生成する.
def generate_log_return(seq):
    # 株価のシミュレーションでseqを生み出すと負になることがあるのでその時はエラーを吐く．
    if np.any(seq <= 0):
        raise ValueError("no way! youre so unlucky!")
    else:
        log_returns = np.diff(np.log(seq))
        return log_returns
    
#-----------------------#
#各種シミュレーション用の関数#
#-----------------------#

# winodow size Nでのmoving window methodを適用する関数
def MW_method(seq, N):
    cumsum_seq = np.insert(np.cumsum(seq),0,0)
    moving_window_seq = (cumsum_seq[N:] - cumsum_seq[:-N])
    return moving_window_seq

# 分散の推定量を計算する関数
def variance_estimator(seq,N):
    sum1 = MW_method(seq**2, N)
    sum2 = MW_method(seq, N)
    S2 = (sum1 - (sum2**2)/N)/(N-1)
    return S2

#---------------------#
#2.1節用のシミュレーション#
#---------------------#

seq1 = std_normal_iid(300)
seq2 = std_normal_iid(300)
seq3 = std_normal_iid(300)

Nlist = [5, 25, 50]

# どういったサンプルか確認するためのプロット
plt.figure(figsize=(10,6))
plt.plot(seq1, label='sample path 1')
plt.plot(seq2, label='sample path 2')
plt.plot(seq3, label='sample path 3')
plt.title(f'Sample paths of standard normal iid')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.savefig('standard_normal_iid_sample_paths.png')
plt.show()

# window sizeごとのmoving window methodの適用結果確認するプロット．
for N in Nlist:
    mw_seq1 = MW_method(seq1, N)
    mw_seq2 = MW_method(seq2, N)
    mw_seq3 = MW_method(seq3, N)
    
    plt.figure(figsize=(10,6))
    plt.plot(mw_seq1, label='sample path 1')
    plt.plot(mw_seq2, label='sample path 2')
    plt.plot(mw_seq3, label='sample path 3')
    plt.title(f'Moving Window Method with window size {N}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid()
    plt.savefig(f'moving_window_N_{N}_sample_paths.png')
    plt.show()
    

#パスを1つに絞り，window sizeごとの平均収益率の挙動を比較するためのプロット．

# 元のサンプルパスのプロット
plt.figure(figsize=(10,6))
plt.plot(seq1, label='sample path 1')
plt.title(f'Sample paths of standard normal iid')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.savefig('standard_normal_iid_sample_path1.png')
plt.show()  

plt.figure(figsize=(10,6))
for N in Nlist:
    mean_mw_seq1 = MW_method(seq1, N)/N
    data_for_plot = np.append([0] * N,mean_mw_seq1)
    plt.plot(data_for_plot, label='N = '+ str(N))
    
plt.title(f'Moving Window Method with various window sizes')
plt.xlabel('Time')
plt.ylabel('Mean of Cumulative Log Return')
plt.legend()
plt.grid()
plt.savefig('moving_window_mean_comparison.png')
plt.show()

# 各window sizeごとの分散の推定量のプロット

plt.figure(figsize=(10,6))
for N in Nlist:
    variance_seq1 = variance_estimator(seq1, N)
    variance_for_plot = np.append([0] * N,variance_seq1)
    plt.plot(variance_for_plot, label='N = '+ str(N))
    
plt.title(f'Variance Estimator with various window sizes')
plt.xlabel('Time')
plt.ylabel('Variance Estimator')
plt.legend()
plt.grid()
plt.savefig('variance_estimator_comparison.png')
plt.show()

#---------------------#
#2.2節用のシミュレーション#
#---------------------#
# T = 1年とし，日次データ(M=252)を考える
# 各種パラメーターはS0=100，金利r=0.05，ボラティリティsigma=0.2とする．

seq1 = stock_price(100, 1, 0.05, 0.2, 252)
seq2 = stock_price(100, 1, 0.05, 0.2, 252)
seq3 = stock_price(100, 1, 0.05, 0.2, 252)

plt.figure(figsize=(10,6))
plt.plot(seq1, label='sample path 1')
plt.plot(seq2, label='sample path 2')
plt.plot(seq3, label='sample path 3')
plt.title(f'Sample paths of stock prices')
plt.xlabel('Time')
plt.ylabel('stock price')
plt.legend()
plt.grid()
plt.savefig('stock_price_sample_paths.png')
plt.show()

log_return_seq1 = generate_log_return(seq1)
log_return_seq2 = generate_log_return(seq2)
log_return_seq3 = generate_log_return(seq3)

plt.figure(figsize=(10,6))
plt.plot(log_return_seq1, label='sample path 1')
plt.plot(log_return_seq2, label='sample path 2')
plt.plot(log_return_seq3, label='sample path 3')
plt.title(f'Cumulative log returns of stock prices')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid()
plt.savefig('log_return_sample_paths.png')
plt.show()

Nlist = [5, 25, 50]

for N in Nlist:
    mw_log_return_seq1 = MW_method(log_return_seq1, N)
    mw_log_return_seq2 = MW_method(log_return_seq2, N)
    mw_log_return_seq3 = MW_method(log_return_seq3, N)
    
    plt.figure(figsize=(10,6))
    plt.plot(mw_log_return_seq1, label='sample path 1')
    plt.plot(mw_log_return_seq2, label='sample path 2')
    plt.plot(mw_log_return_seq3, label='sample path 3')
    plt.title(f'Moving Window Method with window size {N}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid()
    plt.savefig(f'stock_price_moving_window_N_{N}_sample_paths.png')
    plt.show()
    
# パスを1つに絞り，window sizeごとの分散の推定量の挙動を比較するためのプロット．
# 元のサンプルパスのプロット
plt.figure(figsize=(10,6))
plt.plot(log_return_seq1, label='sample path 1')
plt.title(f'Cumulative log returns of stock prices')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid()
plt.savefig('stock_price_log_return_sample_path1.png')
plt.show() 

plt.figure(figsize=(10,6))
for N in Nlist:
    mean_mw_log_return_seq1 = MW_method(log_return_seq1, N)/N
    data_for_plot = np.append([0] * N,mean_mw_log_return_seq1)
    plt.plot(data_for_plot, label='N = '+ str(N))

plt.title(f'Moving Window Method with various window sizes')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid()
plt.savefig('stock_price_moving_window_mean_comparison.png')
plt.show()

#---------------------#
#2.3節用のシミュレーション#
#---------------------#
# 株価のボラティリティが途中で変化する場合を考える．
# 描写のわかりやすさのため2年おきにボラティリティが変化するようにする．
first_years = stock_price(1000, 2, 0.05, 0.2, 504)
second_years = stock_price(first_years[-1], 2, 0.05, 0.5, 504)
third_years = stock_price(second_years[-1], 2, 0.05, 0.1, 504)

stock_prices = np.concatenate([first_years, second_years[1:], third_years[1:]])

plt.figure(figsize=(10,6))
plt.plot(stock_prices, label='stock price path')
plt.plot(503,stock_prices[503], marker = '*', color = 'red', label = "volatility change point")
plt.plot(1007,stock_prices[1007], marker = '*', color = 'red')
plt.title(f'Sample path of stock prices with changing volatility')
plt.xlabel('Time')
plt.ylabel('stock price')
plt.legend()
plt.grid()
plt.savefig('changing_volatility_stock_price_path.png')
plt.show()

# 対数収益率過程の生成
log_return_seq = generate_log_return(stock_prices)
plt.figure(figsize=(10,6))
plt.plot(log_return_seq, label='cumulative log return path')
plt.plot(503,log_return_seq[503], marker = '*', color = 'red', label = "volatility change point")
plt.plot(1007,log_return_seq[1007], marker = '*', color = 'red')
plt.title(f'Cumulative log returns of stock prices with changing volatility')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid()
plt.savefig('changing_volatility_log_return_path.png')
plt.show()

Nlist = [5, 50, 100, 300]

for N in Nlist:
    mw_log_return_seq = np.append([0]*N,MW_method(log_return_seq, N))
    
    plt.figure(figsize=(10,6))
    plt.plot(mw_log_return_seq, label='cumulative log return path')
    plt.plot(503,mw_log_return_seq[503], marker = '*', color = 'red', label = "volatility change point")
    plt.plot(1007,mw_log_return_seq[1007], marker = '*', color = 'red')
    plt.title(f'Moving Window Method with window size {N} for changing volatility case')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid()
    plt.savefig(f'changing_volatility_moving_window_N_{N}_path.png')
    plt.show()