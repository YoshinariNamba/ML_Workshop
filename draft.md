機械学習講座1 - 線形回帰モデル -
================
Yoshinari Namba
2021/01/12

# 0. イントロダクション

### 今日のゴール

-   **実証経済学と機械学習のアプローチの違いを理解する**

インターン先で機械学習に触れた際に経済学との違いを痛感しました．  
今日の講義ではその感覚を皆さんに共有できればうれしいです！

### タイムテーブル

-   \~15:20: 機械学習とは  
-   \~15:40: ハンズオン1

# 1. 機械学習とは

### データ分析

-   モデリングを伴わない分析
    -   記述統計，可視化 etc.  
-   モデリングを伴う分析
    -   **推論 (inference)**:
        観測可能な![X](https://latex.codecogs.com/png.latex?X "X")と![Y](https://latex.codecogs.com/png.latex?Y "Y")の関係性を推測する  
    -   **予測 (prediction)**:
        観測可能な変数![X](https://latex.codecogs.com/png.latex?X "X")から観測不能な![Y](https://latex.codecogs.com/png.latex?Y "Y")を予測する

### 機械学習とは

-   データから変数間の複雑な構造を学習する
    -   経済学: 経済学理論が変数間のルールを決める(theory-driven)  
    -   機械学習:
        アルゴリズムがデータから最良のルールを導く(data-driven)  
-   「予測」が主な関心
    -   経済学: パラメータ推定値
        (![\\hat{\\beta}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta%7D "\hat{\beta}"))
        に関心がある  
    -   機械学習: 予測値
        (![\\hat{y}](https://latex.codecogs.com/png.latex?%5Chat%7By%7D "\hat{y}"))
        に関心がある

### 教師あり機械学習の実行方法

機械学習は主に**教師あり学習**，**教師なし学習**，**強化学習**の3種類に大別されます．  
今日はその中でも**教師あり学習**を扱います．教師あり学習とはざっくり言うと「データの中に予測したい変数の実測値が含まれている」場合の学習です．

#### ステップ

この演習では次の3つのステップで教師あり機械学習を実行します．  
1. **学習**  
- <u>データ</u>からパラメータを推定してモデルを構築  
- Rでは`lm()`, `glm()`, `glmnet::glmnet()`など  
2. **予測**  
- 構築したモデルに<u>新しいデータ</u>を入力して予測値を算出  
- Rでは`predict()`など  
3. **モデル評価**  
-
実測値と予測値のズレを予測誤差と言います．予測誤差が小さいほど優れたモデルと言えそうです．  
- 連続変数の予測誤差は多くの場合 **平均二乗誤差(Mean Squared Error;
MSE)** で計測します．  

![
  \\text{MSE}=\\frac{1}{N}\\sum\_{i=1}^{N}(y_i-\\hat{y}\_i)^2
](https://latex.codecogs.com/png.latex?%0A%20%20%5Ctext%7BMSE%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i-%5Chat%7By%7D_i%29%5E2%0A "
  \text{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
")

- ![N](https://latex.codecogs.com/png.latex?N "N"): サンプル数  
- ![y_i](https://latex.codecogs.com/png.latex?y_i "y_i"): 実測値  
-
![\\hat{y}\_i](https://latex.codecogs.com/png.latex?%5Chat%7By%7D_i "\hat{y}_i"):
予測値

#### ポイント

教師あり学習には**2種類のデータセットが必要です**．  
- ステップ1で<u>パラメータを推定する</u>ために使うデータ  
- ステップ2で<u>予測値を算出する</u>ためにモデルに入力するデータ

手元のデータセットを **学習データ(training data)** と
**テストデータ(test data)** に分割して モデルを評価する方法を
**交差検証(Cross-Validation)** と言います．  
「同じデータ使えばよくない？」と思うかもしれませんが，学習と予測で同じデータセットを使うと予測精度をうまく評価できなくなります．理由は後ほど説明しますね．

# 2. ハンズオン1: OLSを使って予測

上記のステップに従って，まずは最小2乗法を使って教師あり学習を実行してみましょう！  
町のデータを使って住宅価格を予測したいケースを考えます．モチベーションはあくまで
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

`{MASS}`パッケージに入っているボストンの住宅価格データを使います．詳細は[こちらのサイト](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)ないしは`?MASS::Boston`で見れます．

``` r
## data source
data("Boston", package = "MASS")
```

-   観察単位: 町
-   変数 (`?MASS::Boston`参照)  
    – medv: median value of owner-occupied homes in $1000s (目的変数)  
    – crim: per capita crime rate by town  
    – zn: proportion of residential land zone for lots over 25,000
    sq.ft.  
    – indus: proportion of non-retail business acre per town  
    – chas: Charles River dummy variable  
    – nox: nitrogen oxides concentration  
    – rm: average room per dwelling  
    – age: proportion of owner-occupied units built prior to 1940  
    – dis: weighed mean of distances to five Boston employment centers  
    – rad: index of accessibility to radial highways  
    – tax: full-value property-tax rate per $10,000  
    – ptratio: pupil-teacher ratio by town  
    – black:
    ![1000(Bk - 0.63)^2](https://latex.codecogs.com/png.latex?1000%28Bk%20-%200.63%29%5E2 "1000(Bk - 0.63)^2")
    where Bk is the proportion of blacks by town  
    – lstat: lower status of the population (percent)

記述統計を見てみましょう．

``` r
Boston %>% stargazer(type = "text")
```

    ## 
    ## ===============================================================
    ## Statistic  N   Mean   St. Dev.  Min   Pctl(25) Pctl(75)   Max  
    ## ---------------------------------------------------------------
    ## crim      506  3.614   8.602   0.006   0.082    3.677   88.976 
    ## zn        506 11.364   23.322    0       0       12.5     100  
    ## indus     506 11.137   6.860   0.460   5.190    18.100  27.740 
    ## chas      506  0.069   0.254     0       0        0        1   
    ## nox       506  0.555   0.116   0.385   0.449    0.624    0.871 
    ## rm        506  6.285   0.703   3.561   5.886    6.624    8.780 
    ## age       506 68.575   28.149  2.900   45.025   94.075  100.000
    ## dis       506  3.795   2.106   1.130   2.100    5.188   12.126 
    ## rad       506  9.549   8.707     1       4        24      24   
    ## tax       506 408.237 168.537   187     279      666      711  
    ## ptratio   506 18.456   2.165   12.600  17.400   20.200  22.000 
    ## black     506 356.674  91.295  0.320  375.378  396.225  396.900
    ## lstat     506 12.653   7.141   1.730   6.950    16.955  37.970 
    ## medv      506 22.533   9.197     5      17.0      25      50   
    ## ---------------------------------------------------------------

## 2-1. 学習

後で使用する`{glmnet}`パッケージの都合上，予測変数と目的変数をそれぞれ`X`,
`y`とし，別のオブジェクトとして定義します．

``` r
# 予測変数と目的変数別のオブジェクトに格納
X <- Boston[, -14]
y <- Boston$medv
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
  step_poly(-chas) %>% # 2乗項を加える
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
mdl_ols_complex <- lm(data = X_complex_train, formula = y_train ~ .)
```

このケースの目的は予測ですが，一応モデルの説明力を比較してみましょう．

``` r
# 自由度調整済み決定係数の比較
summary(mdl_ols_simple)$adj.r.squared
```

    ## [1] 0.725237

``` r
summary(mdl_ols_complex)$adj.r.squared
```

    ## [1] 0.9136481

## 2-2. 予測

予測には`predict(モデル, データ)`という関数を使用します．先ほど構築したモデルにテストデータを入力して予測値を算出します．

``` r
pred_ols_simple <- predict(mdl_ols_simple, X_test)
pred_ols_complex <- predict(mdl_ols_complex, X_complex_test)
```

    ## Warning in predict.lm(mdl_ols_complex, X_complex_test): prediction from a rank-
    ## deficient fit may be misleading

また，モデルに学習データを入力した場合の予測値も比較のために定義しておきましょう．

``` r
pred_ols_simple_train <- predict(mdl_ols_simple, X_train)
pred_ols_complex_train <- predict(mdl_ols_complex, X_complex_train)
```

    ## Warning in predict.lm(mdl_ols_complex, X_complex_train): prediction from a rank-
    ## deficient fit may be misleading

## 2-3. モデル評価

各モデルの予測評価を評価するために，平均二乗誤差(MSE)を算出します

``` r
# 平均二乗誤差
## テストデータの予測誤差
mse_ols_simple <- sum( ( y_test - pred_ols_simple )^2 ) / length(y_test)
mse_ols_complex <- sum( ( y_test - pred_ols_complex )^2 ) / length(y_test)

## 学習データの予測誤差
mse_ols_simple_train <- sum( ( y_train - pred_ols_simple_train )^2 ) / length(y_train)
mse_ols_complex_train <- sum( ( y_train - pred_ols_complex_train )^2 ) / length(y_train)
```

予測誤差を比較してみましょう．

``` r
# 結果の要約
rslt_mse_ols <- data.frame(simple = c(mse_ols_simple_train, mse_ols_simple), 
                           complex = c(mse_ols_complex_train, mse_ols_complex))
rownames(rslt_mse_ols) <- c("training", "test")

rslt_mse_ols
```

    ##            simple   complex
    ## training 21.13066  3.821487
    ## test     25.89717 16.404760

# 3. 汎化誤差

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

確率変数が![D](https://latex.codecogs.com/png.latex?D "D")と![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")の2種類あることに注意してください．  
ノイズ![\\sigma^2](https://latex.codecogs.com/png.latex?%5Csigma%5E2 "\sigma^2")はどんなモデルでも除外できない誤差です．モデリングの工夫によって汎化誤差を縮小するにはバイアスとバリアンスの和を縮小する必要があります．  
バイアスは「真のモデル」と「学習したモデル」のズレを表します．(図解)
バリアンスは学習モデルで予測したときの予測値の分散を表します．
複雑なモデルではバイアスは縮小する一方でバリアンスが拡大します．

## 3-2. 罰則付き回帰モデル

ほどよい「複雑さ」を見つける方法の1つに
**罰則付き回帰モデル(正則化回帰モデル)** があります．  
次のような線形回帰モデルで予測するとき，

![
\\begin{align}
  y 
  &= \\beta_0 + \\beta_1X_1 + \\cdots + \\beta_pX_p + \\epsilon
\\end{align}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Balign%7D%0A%20%20y%20%0A%20%20%26%3D%20%5Cbeta_0%20%2B%20%5Cbeta_1X_1%20%2B%20%5Ccdots%20%2B%20%5Cbeta_pX_p%20%2B%20%5Cepsilon%0A%5Cend%7Balign%7D%0A "
\begin{align}
  y 
  &= \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p + \epsilon
\end{align}
")

最小二乗法ではデータの
**誤差(![y_i - \\hat{y}\_i](https://latex.codecogs.com/png.latex?y_i%20-%20%5Chat%7By%7D_i "y_i - \hat{y}_i"))の2乗和**
を最小化します．一方で，罰則付き回帰モデルでは残差二乗和にパラメータの大きさ(2乗
or 絶対値)を足したものを最小化します．

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

この罰則(![\\lambda\\sum\_{k=0}^p\\beta_k^2](https://latex.codecogs.com/png.latex?%5Clambda%5Csum_%7Bk%3D0%7D%5Ep%5Cbeta_k%5E2 "\lambda\sum_{k=0}^p\beta_k^2")または![\\lambda\\sum\_{k=0}^p\|\\beta_k\|](https://latex.codecogs.com/png.latex?%5Clambda%5Csum_%7Bk%3D0%7D%5Ep%7C%5Cbeta_k%7C "\lambda\sum_{k=0}^p|\beta_k|"))を目的関数に含めることで，![y](https://latex.codecogs.com/png.latex?y "y")の予測にあまり役に立たない変数のパラメータは小さく推定されます．  
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

## 4-2. 予測

``` r
## 予測
pred_ridge <- predict(mdl_ridge, as.matrix(X_complex_test))
pred_lasso <- predict(mdl_lasso, as.matrix(X_complex_test))
```

`glmnet`の予測変数は`matrix`型に直さないとエラーになるようです．

## 4-3. モデル評価

``` r
## モデル評価
mse_ridge <- sum( (y_test - as.vector(pred_ridge))^2 ) / length(y_test)
mse_lasso <- sum( (y_test - as.vector(pred_lasso))^2 ) / length(y_test)

## 結果の要約
rslt_mse_penalized <- data.frame(OLS = mse_ols_complex, Ridge = mse_ridge, Lasso = mse_lasso)
rownames(rslt_mse_penalized) <- "MSE"

rslt_mse_penalized
```

    ##          OLS    Ridge   Lasso
    ## MSE 16.40476 25.84726 19.1522
