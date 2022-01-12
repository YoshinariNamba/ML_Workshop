機械学習講座1 - 線形回帰モデル -
================
Yoshinari Namba
2021/01/12

# 0. イントロダクション

### 今日のゴール

-   **実証経済学と機械学習のアプローチの違いを理解する**

インターン先で機械学習に触れた際に経済学との違いを痛感しました．  
その感覚を皆さんに共有できればと思います！

### タイムテーブル

-   \~15:25: 1. 機械学習とは
-   \~15:40: 2. ハンズオン1
-   \~16:15: 3. 汎化誤差の対処
-   \~16:45: 4. ハンズオン2

# 1. 機械学習とは

### データ分析

-   モデリングを伴わない分析
    -   記述統計，可視化 etc.  
-   モデリングを伴う分析
    -   **推論 (inference)**:
        ![X](https://latex.codecogs.com/png.latex?X "X")と![Y](https://latex.codecogs.com/png.latex?Y "Y")の関係性を推測する  
    -   **予測 (prediction)**:
        ![X](https://latex.codecogs.com/png.latex?X "X")を使って![Y](https://latex.codecogs.com/png.latex?Y "Y")を予測する

### 機械学習とは

-   アルゴリズムがデータから最良のルールを導く(data-driven)
    -   経済学: 経済学理論が変数間のルールを決める(theory-driven)  
-   主な関心は予測 → 予測値
    (![\\hat{y}](https://latex.codecogs.com/png.latex?%5Chat%7By%7D "\hat{y}"))
    に関心がある
    -   経済学: パラメータ推定値
        (![\\hat{\\beta}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta%7D "\hat{\beta}"))
        に関心がある

### 教師あり機械学習の実行方法

機械学習は主に**教師あり学習**，**教師なし学習**，**強化学習**の3種類に大別されます．  
今日はその中でも**教師あり学習**を扱います．教師あり学習とはざっくり言うと「データの中に予測したい変数の実測値が含まれている」場合の学習です．

#### ステップ

この演習では次の3つのステップで教師あり機械学習を実行します．

1.  **学習**
    -   <u>データ</u>からパラメータを推定してモデルを構築
    -   Rでは`lm()`, `glm()`, `glmnet::glmnet()`など
2.  **予測**
    -   構築したモデルに<u>新しいデータ</u>を入力して予測値を算出
    -   Rでは`predict()`など
3.  **モデル評価**
    -   実測値と予測値のズレを予測誤差と言います．予測誤差が小さいほど優れたモデルです
    -   連続変数の予測誤差は多くの場合 **平均二乗誤差(Mean Squared
        Error; MSE)** で計測します．

#### 予測誤差

**平均二乗誤差(MSE)** は次のように定義されます．

![
  \\text{MSE}=\\frac{1}{N}\\sum\_{i=1}^{N}(y_i-\\hat{y}\_i)^2
](https://latex.codecogs.com/png.latex?%0A%20%20%5Ctext%7BMSE%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i-%5Chat%7By%7D_i%29%5E2%0A "
  \text{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
")

-   ![N](https://latex.codecogs.com/png.latex?N "N"): サンプル数
-   ![y_i](https://latex.codecogs.com/png.latex?y_i "y_i"): 実測値
-   ![\\hat{y}\_i](https://latex.codecogs.com/png.latex?%5Chat%7By%7D_i "\hat{y}_i"):
    予測値

#### ポイント

教師あり学習には**2種類のデータセットが必要です**．  
- ステップ1で<u>パラメータを推定する</u>ために使うデータ  
- ステップ2で<u>予測値を算出する</u>ためにモデルに入力するデータ

手元のデータセットを **学習データ(training data)** と
**テストデータ(test data)** に分割して モデルを評価する方法を
**交差検証(Cross-Validation)** と言います．  
「同じデータを使えばよくない？」と思うかもしれませんが，学習と予測で同じデータセットを使うと予測精度をうまく評価できなくなります．理由は後ほど説明しますね．

# 2. ハンズオン1: OLSを使って予測

上記のステップに従って，まずは最小2乗法を使って教師あり学習を実行してみましょう！  
大学のデータを使って出願者数を予測したいケースを考えます．モチベーションはあくまで
**予測** であって， **変数の間の関係性には関心がない状況**
を仮定します．

## 2-0. 準備

### パッケージの読み込み

今日使用するパッケージは以下の4つです．インストールが済んでいない方は`install.packages("パッケージ名")`から実行してください．

``` r
## libary
library(tidyverse)
library(tidymodels)
library(glmnet)
library(stargazer)
```

### データ

大学の出願状況データを使います．元データの詳細は[こちらのp5](https://cran.r-project.org/web/packages/ISLR/ISLR.pdf)を参照してください．

``` r
## data source
df <- read.csv("College_editted.csv")
```

-   観察単位: 大学
-   変数
    -   `Private`: 私立ダミー
    -   `Apps`: 出願者数 **(目的変数)**
    -   `Accept`: 合格者数
    -   `Enroll`: 入学者数
    -   `Top10perc`: 上位10％の高校出身者の割合
    -   `Top25perc`: 上位25％の高校出身者の割合
    -   `F.Undergrad`: 全日制の学部生の人数
    -   `P.Undergrad`: 定時制の学部生の人数
    -   `Outstate`: 州外の学生にかかる追加授業料
    -   `Room.Board`: 学生寮の費用
    -   `Books`: 教科書の推定費用
    -   `Personal`: 推定個人費用
    -   `PhD`: 博士課程の学科の割合
    -   `Terminal`: 最終学位の学科の割合
    -   `S.F.Ratio`: 学生/学科の割合
    -   `perc.alumni`: 寄付している卒業生の割合
    -   `Expend`: 学生あたりの指導費用
    -   `Grad.Rate`: 卒業率

記述統計を見てみましょう．

``` r
df %>% stargazer(type = "text")
```

    ## 
    ## ===================================================================
    ## Statistic    N     Mean    St. Dev.   Min  Pctl(25) Pctl(75)  Max  
    ## -------------------------------------------------------------------
    ## Private     777   0.727      0.446     0      0        1       1   
    ## Apps        777 3,001.638  3,870.201  81     776     3,624   48,094
    ## Accept      777 2,018.804  2,451.114  72     604     2,424   26,330
    ## Enroll      777  779.973    929.176   35     242      902    6,392 
    ## Top10perc   777   27.559    17.640     1      15       35      96  
    ## Top25perc   777   55.797    19.805     9      41       69     100  
    ## F.Undergrad 777 3,699.907  4,850.421  139    992     4,005   31,643
    ## P.Undergrad 777  855.299   1,522.432   1      95      967    21,836
    ## Outstate    777 10,440.670 4,023.016 2,340  7,320    12,925  21,700
    ## Room.Board  777 4,357.526  1,096.696 1,780  3,597    5,050   8,124 
    ## Books       777  549.381    165.105   96     470      600    2,340 
    ## Personal    777 1,340.642   677.071   250    850     1,700   6,800 
    ## PhD         777   72.660    16.328     8      62       85     103  
    ## Terminal    777   79.703    14.722    24      71       92     100  
    ## S.F.Ratio   777   14.090     3.958   2.500  11.500   16.500  39.800
    ## perc.alumni 777   22.744    12.392     0      13       31      64  
    ## Expend      777 9,660.171  5,221.768 3,186  6,751    10,830  56,233
    ## Grad.Rate   777   65.463    17.178    10      53       78     118  
    ## -------------------------------------------------------------------

## 2-1. 学習

後で使用する`{glmnet}`パッケージの都合上，予測変数と目的変数をそれぞれ`X`,
`y`とし，別のオブジェクトとして定義します．

``` r
# 予測変数と目的変数別のオブジェクトに格納
X <- df[, -2]
y <- df$Apps
```

データセットをランダムに学習データとテストデータに分割します．

``` r
set.seed(20220112)
N_train <- round(length(y)*0.8) #  学習データの観測数 
id_train <- sample(x = length(y), size = N_train, replace = FALSE) # 学習データのID

# 学習データ
X_train <- X[id_train, ]
y_train <- y[id_train]

# テストデータ
X_test <- X[-id_train, ]
y_test <- y[-id_train]
```

### 2-1-1. シンプルなモデル

まずはシンプルにモデルを構築します．

``` r
mdl_ols_simple <- lm(data = X_train, formula = y_train ~ .)
```

ここで，引数`formula =`の右辺`.`は`data =`で指定したデータセットのすべての変数を予測変数に使用することを命令する演算子です．

### 2-1-2. 複雑なモデル

次に複雑なモデルを構築してみましょう．交差項と2乗項を予測変数に含めてモデリングしてみます．  
まずは予測変数の交差項と2乗項を加えたデータ`X_complex_train`,
`X_complex_test`を作成します．交差項と2乗項の作成には`recipes::recipe()`という関数を使います．

``` r
## 予測変数の交差項と2乗項を新たな予測変数として作成
X_complex <- recipe(~ ., data = X) %>%
  step_interact(~all_predictors():all_predictors()) %>% # 交差項を加える
  step_poly(-Private) %>% # 2乗項を加える
  prep() %>% 
  bake(X) 

# 学習データ
X_complex_train <- X_complex[id_train, ]

# テストデータ
X_complex_test <- X_complex[-id_train, ]
```

各自交差項や2乗項が追加されていることを確認してみてください．  
学習データを使って複雑モデルverのOLSを実行し，モデルを`mdl_ols_complex`と名付けます．

``` r
mdl_ols_complex <- 
```

このケースの目的は予測ですが，一応モデルの説明力を比較してみましょう．

``` r
# 自由度調整済み決定係数の比較
summary(mdl_ols_simple)$adj.r.squared; summary(mdl_ols_complex)$adj.r.squared
```

    ## [1] 0.9300762

    ## [1] 0.9876334

## 2-2. 予測

予測には`predict(モデル, データ)`という関数を使用します．先ほど構築したモデルにテストデータを入力して予測値を算出します．

``` r
pred_ols_simple <- predict(mdl_ols_simple, X_test)
pred_ols_complex <- predict(mdl_ols_complex, X_complex_test)
```

また，モデルに学習データを入力した場合の予測値も比較のために定義しておきましょう．

``` r
pred_ols_simple_train <- 
pred_ols_complex_train <- 
```

## 2-3. モデル評価

各モデルの予測評価を評価するために，平均二乗誤差(MSE)を算出します

``` r
# 平均二乗誤差
## テストデータの予測誤差
mse_ols_simple <- sum( ( y_test - pred_ols_simple )^2 ) / length(y_test)
mse_ols_complex <- sum( ( y_test - pred_ols_complex )^2 ) / length(y_test)

## 学習データの予測誤差
mse_ols_simple_train <- 
mse_ols_complex_train <- 
```

予測誤差を比較してみましょう．

``` r
# 結果の要約
rslt_mse_ols <- data.frame(simple = c(mse_ols_simple_train, mse_ols_simple), 
                           complex = c(mse_ols_complex_train, mse_ols_complex))
rownames(rslt_mse_ols) <- c("training", "test")

rslt_mse_ols
```

    ##           simple    complex
    ## training 1020740   94447.92
    ## test     1318296 7638216.30

# 3. 汎化誤差の対処

## 3-1. バイアス-バリアンス トレードオフ

計量経済学では，いくつかの仮定の下で最小2乗推定量は不偏性と一致性を満たすことを習いました．パラメータ推定値がバイアスしないということです．  
では，「バイアスが小さい =
予測精度が高い」ということなのでしょうか?実はそうではないようです．

### 予測誤差の期待値の分解

![y,X](https://latex.codecogs.com/png.latex?y%2CX "y,X")の間に![y=f(X)+\\epsilon](https://latex.codecogs.com/png.latex?y%3Df%28X%29%2B%5Cepsilon "y=f(X)+\epsilon")という関係があると仮定し(![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")は平均![0](https://latex.codecogs.com/png.latex?0 "0"),
分散![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2")の誤差)．データ![D](https://latex.codecogs.com/png.latex?D "D")から![f(X)](https://latex.codecogs.com/png.latex?f%28X%29 "f(X)")を近似する![\\hat{f}(X; D)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28X%3B%20D%29 "\hat{f}(X; D)")というモデルを構築して新たな![X](https://latex.codecogs.com/png.latex?X "X")から![y](https://latex.codecogs.com/png.latex?y "y")を予測する状況を考えます．  
このとき予測誤差![y-\\hat{f}(X; D)](https://latex.codecogs.com/png.latex?y-%5Chat%7Bf%7D%28X%3B%20D%29 "y-\hat{f}(X; D)")の期待値はバイアス(![\\text{Bias}\_D(.)^2](https://latex.codecogs.com/png.latex?%5Ctext%7BBias%7D_D%28.%29%5E2 "\text{Bias}_D(.)^2"))とバリアンス(![\\text{Var}\_{D}(.)](https://latex.codecogs.com/png.latex?%5Ctext%7BVar%7D_%7BD%7D%28.%29 "\text{Var}_{D}(.)")),
ノイズ(![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2"))に分解できます．

![
\\begin{align\*}
  \\text{E}\_{D, \\epsilon }\\left\[(y - \\hat{f}(X ; D))^2 \\right\] 
  &= \\left\[ \\text{Bias}\_{D} ( \\hat{f} (X ; D) ) \\right\]^2 +  \\text{Var}\_{D} \\left\[ \\hat{f} (X ; D) \\right\] + \\sigma^2
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20%5Ctext%7BE%7D_%7BD%2C%20%5Cepsilon%20%7D%5Cleft%5B%28y%20-%20%5Chat%7Bf%7D%28X%20%3B%20D%29%29%5E2%20%5Cright%5D%20%0A%20%20%26%3D%20%5Cleft%5B%20%5Ctext%7BBias%7D_%7BD%7D%20%28%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%29%20%5Cright%5D%5E2%20%2B%20%20%5Ctext%7BVar%7D_%7BD%7D%20%5Cleft%5B%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%5Cright%5D%20%2B%20%5Csigma%5E2%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  \text{E}_{D, \epsilon }\left[(y - \hat{f}(X ; D))^2 \right] 
  &= \left[ \text{Bias}_{D} ( \hat{f} (X ; D) ) \right]^2 +  \text{Var}_{D} \left[ \hat{f} (X ; D) \right] + \sigma^2
\end{align*}
")

ここで，

![
\\begin{align\*}
  \\text{Bias}\_{D} ( \\hat{f} (X ; D) ) 
  &= \\text{E}\_{D}(\\hat{f}(X;D)) - f(X) \\\\
  \\text{Var}\_{D} \\left\[ \\hat{f} (X ; D) \\right\]
  &= \\text{E}\_{D} \\left\[ ( \\text{E}\_{D}\[\\hat{f}(X; D)\] - \\hat{f}(X; D) )^2 \\right\].
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20%5Ctext%7BBias%7D_%7BD%7D%20%28%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%29%20%0A%20%20%26%3D%20%5Ctext%7BE%7D_%7BD%7D%28%5Chat%7Bf%7D%28X%3BD%29%29%20-%20f%28X%29%20%5C%5C%0A%20%20%5Ctext%7BVar%7D_%7BD%7D%20%5Cleft%5B%20%5Chat%7Bf%7D%20%28X%20%3B%20D%29%20%5Cright%5D%0A%20%20%26%3D%20%5Ctext%7BE%7D_%7BD%7D%20%5Cleft%5B%20%28%20%5Ctext%7BE%7D_%7BD%7D%5B%5Chat%7Bf%7D%28X%3B%20D%29%5D%20-%20%5Chat%7Bf%7D%28X%3B%20D%29%20%29%5E2%20%5Cright%5D.%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  \text{Bias}_{D} ( \hat{f} (X ; D) ) 
  &= \text{E}_{D}(\hat{f}(X;D)) - f(X) \\
  \text{Var}_{D} \left[ \hat{f} (X ; D) \right]
  &= \text{E}_{D} \left[ ( \text{E}_{D}[\hat{f}(X; D)] - \hat{f}(X; D) )^2 \right].
\end{align*}
")

確率変数が ![D](https://latex.codecogs.com/png.latex?D "D") と
![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")
の2種類あることに注意してください．![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")
は平均 ![0](https://latex.codecogs.com/png.latex?0 "0") ，分散
![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma")
の仮定と ![X](https://latex.codecogs.com/png.latex?X "X")
と独立であることから期待値を取ると省かれます．  
データ![D](https://latex.codecogs.com/png.latex?D "D")が確率変数であるため，パラメータ推定に使用したデータでの誤差と新たなデータ(out-of-sample)を使って予測する場合の誤差が異なります．後者は汎化誤差(generalization
error)と言います．

ノイズ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2")はどんなモデルでも除外できない誤差です．モデリングの工夫によって汎化誤差を縮小するにはバイアスとバリアンスの和を縮小する必要があります．  
バイアスは「真のモデル」と「学習したモデル」のズレを表します．
バリアンスは学習モデルで予測したときの予測値の分散を表します．
複雑なモデルではバイアスは縮小する一方でバリアンスが拡大します．

## 3-2. 罰則付き回帰モデル

ほどよい「複雑さ」を見つける方法の1つに
**罰則付き回帰モデル(正則化回帰モデル)** があります．  
次のような線形回帰モデルで予測するとき，

![
\\begin{align}
  y 
  &= \\beta_0 + \\beta_1X_1 + \\cdots + \\beta_pX_p + \\epsilon \\nonumber 
\\end{align}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%7D%0A%20%20y%20%0A%20%20%26%3D%20%5Cbeta_0%20%2B%20%5Cbeta_1X_1%20%2B%20%5Ccdots%20%2B%20%5Cbeta_pX_p%20%2B%20%5Cepsilon%20%5Cnonumber%20%0A%5Cend%7Balign%7D%0A "
\begin{align}
  y 
  &= \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p + \epsilon \nonumber 
\end{align}
")

最小二乗法ではデータの
**誤差(![y_i - \\hat{y}\_i](https://latex.codecogs.com/png.latex?y_i%20-%20%5Chat%7By%7D_i "y_i - \hat{y}_i"))の2乗和**
を最小化します．罰則付き回帰モデルでは残差二乗和に**パラメータの2乗や絶対値**を足したものを最小化します．

![
\\begin{align\*}
  \\text{OLS} &: \\min\_{\\beta} \\  \\sum\_{i = 1}^N (y_i - \\sum\_{k = 0}^p \\beta_k x\_{ik})^2 \\\\
  \\text{Ridge} &: \\min\_{\\beta} \\  \\sum\_{i = 1}^N (y_i - \\sum\_{k = 0}^p \\beta_k x\_{ik})^2  + \\lambda \\sum\_{k = 0}^p \\beta_k^2 \\\\
  \\text{Lasso} &: \\min\_{\\beta} \\  \\sum\_{i = 1}^N (y_i - \\sum\_{k = 0}^p \\beta_k x\_{ik})^2 +  \\lambda \\sum\_{k = 0}^p \| \\beta_k \|
\\end{align\*}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%2A%7D%0A%20%20%5Ctext%7BOLS%7D%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%5Csum_%7Bi%20%3D%201%7D%5EN%20%28y_i%20-%20%5Csum_%7Bk%20%3D%200%7D%5Ep%20%5Cbeta_k%20x_%7Bik%7D%29%5E2%20%5C%5C%0A%20%20%5Ctext%7BRidge%7D%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%5Csum_%7Bi%20%3D%201%7D%5EN%20%28y_i%20-%20%5Csum_%7Bk%20%3D%200%7D%5Ep%20%5Cbeta_k%20x_%7Bik%7D%29%5E2%20%20%2B%20%5Clambda%20%5Csum_%7Bk%20%3D%200%7D%5Ep%20%5Cbeta_k%5E2%20%5C%5C%0A%20%20%5Ctext%7BLasso%7D%20%26%3A%20%5Cmin_%7B%5Cbeta%7D%20%5C%20%20%5Csum_%7Bi%20%3D%201%7D%5EN%20%28y_i%20-%20%5Csum_%7Bk%20%3D%200%7D%5Ep%20%5Cbeta_k%20x_%7Bik%7D%29%5E2%20%2B%20%20%5Clambda%20%5Csum_%7Bk%20%3D%200%7D%5Ep%20%7C%20%5Cbeta_k%20%7C%0A%5Cend%7Balign%2A%7D%0A "
\begin{align*}
  \text{OLS} &: \min_{\beta} \  \sum_{i = 1}^N (y_i - \sum_{k = 0}^p \beta_k x_{ik})^2 \\
  \text{Ridge} &: \min_{\beta} \  \sum_{i = 1}^N (y_i - \sum_{k = 0}^p \beta_k x_{ik})^2  + \lambda \sum_{k = 0}^p \beta_k^2 \\
  \text{Lasso} &: \min_{\beta} \  \sum_{i = 1}^N (y_i - \sum_{k = 0}^p \beta_k x_{ik})^2 +  \lambda \sum_{k = 0}^p | \beta_k |
\end{align*}
")

![\\beta_k^2](https://latex.codecogs.com/png.latex?%5Cbeta_k%5E2 "\beta_k^2")
や
![\|\\beta_k\|](https://latex.codecogs.com/png.latex?%7C%5Cbeta_k%7C "|\beta_k|")
が罰則です．
この罰則を目的関数に含めることで，予測にあまり役立たない変数のパラメータは小さく推定されます．  
なお，ここでハイパーパラメータ![\\lambda](https://latex.codecogs.com/png.latex?%5Clambda "\lambda")は交差検証によってチューニングされます．

# 4. ハンズオン2: 罰則付き回帰モデルの実装

## 4-1. 学習

リッジ回帰やラッソ回帰には`{glmnet}`パッケージを使用します．今回は`glmnet::cv.glmnet()`を使用します．

``` r
## 学習
mdl_ridge <- cv.glmnet(x = as.matrix(X_complex_train), y = y_train, alpha = 1)
mdl_lasso <- cv.glmnet(x = as.matrix(X_complex_train), y = y_train, alpha = 0)
```

`alpha = 1`と指定するとリッジ回帰を実行し，`alpha = 0`と指定するとラッソ回帰を実行します．
`{glmnet}`の関数は予測変数を`matrix`型に直さないとエラーになるようです．

## 4-2. 予測

``` r
## 予測
pred_ridge <- predict(mdl_ridge, as.matrix(X_complex_test))
pred_lasso <- predict(mdl_lasso, as.matrix(X_complex_test))

## 学習データでも予測
pred_ridge_train <- 
pred_lasso_train <- 
```

## 4-3. モデル評価

``` r
## モデル評価
mse_ridge <- sum( (y_test - as.vector(pred_ridge))^2 ) / length(y_test)
mse_lasso <- sum( (y_test - as.vector(pred_lasso))^2 ) / length(y_test)

mse_ridge_train <- sum( (y_train - as.vector(pred_ridge_train))^2 ) / length(y_train)
mse_lasso_train <- sum( (y_train - as.vector(pred_lasso_train))^2 ) / length(y_train)
```

結果を見てみます．

``` r
## 結果の要約
rslt_mse_penalized <- data.frame(OLS = c(mse_ols_complex_train, mse_ols_complex), 
                                 Ridge = c(mse_ridge_train, mse_ridge), 
                                 Lasso = c(mse_lasso_train, mse_lasso))
rownames(rslt_mse_penalized) <- c("training", "test")

rslt_mse_penalized
```

    ##                 OLS   Ridge   Lasso
    ## training   94447.92 2101714 1972900
    ## test     7638216.30 2246839 2644418

# 参考文献

-   James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013) *An
    Introduction to Statistical Learning with applications in R*,
    <https://www.statlearning.com>, Springer-Verlag, New York
-   Mullainathan, Sendhil and Jann Spiess (2007), “Machine Learning: An
    Applied Econometric Approach”, *Journal of Economic Perspectives*
    Volume 31, Number 2—Spring 2017—Pages 87–106,
    <https://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.31.2.87>
-   Wikipedia, Bias-Variance Trade-off,
    <https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff>
