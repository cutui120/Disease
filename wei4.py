import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, PowerTransformer, RobustScaler, KBinsDiscretizer
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_classif, mutual_info_classif, chi2, VarianceThreshold
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
import os
import warnings
import time
import itertools
from scipy import stats
from scipy.stats import randint, uniform
from collections import defaultdict
import json
from datetime import datetime

# 处理loguniform导入问题
try:
    from scipy.stats import loguniform
except ImportError:
    # 对于较新版本的scipy，loguniform已更名或移动
    try:
        from scipy.stats._distn_infrastructure import rv_frozen
        loguniform = uniform  # 使用近似替代
    except ImportError:
        # 定义简单的loguniform函数
        class loguniform:
            def __init__(self, a, b):
                self.a = a
                self.b = b
            def rvs(self, random_state=None):
                if random_state is not None:
                    np.random.seed(random_state)
                return np.exp(np.random.uniform(np.log(self.a), np.log(self.b)))

# 设置警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置显示选项
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 120)

class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_data():
        """加载训练和测试数据"""
        try:
            train_features = pd.read_excel('z1/Xtrain.xlsx')
            train_labels = pd.read_excel('z1/ytrain.xlsx').values.ravel()
            test_features = pd.read_excel('z1/Xtest.xlsx')
            test_labels = pd.read_excel('z1/ytest.xlsx').values.ravel()
            
            print("数据加载成功:")
            print(f"  训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 个特征")
            print(f"  测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 个特征")
            
            # 保存特征名称
            feature_names = train_features.columns.tolist()
            
            # 转换为numpy数组
            X_train = train_features.values
            X_test = test_features.values
            
            return X_train, train_labels, X_test, test_labels, feature_names
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None, None, None, None, None
    
    @staticmethod
    def load_new_data():
        """加载新数据进行预测"""
        try:
            new_features = pd.read_excel('z1/Xnew.xlsx')
            print(f"新数据集加载成功: {new_features.shape[0]} 个样本, {new_features.shape[1]} 个特征")
            return new_features.values, new_features.columns.tolist()
        except FileNotFoundError:
            print("未找到新数据集文件 'z1/Xnew.xlsx'，跳过新数据预测")
            return None, None
        except Exception as e:
            print(f"新数据集加载失败: {e}")
            return None, None

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """自动化特征工程管道"""
    
    def __init__(self, n_pca_components=0.95, n_ica_components=10, 
                 poly_degree=2, n_clusters=5, create_interactions=True,
                 create_polynomial=True, use_pca=True, use_ica=True,
                 variance_threshold=0.01, n_stat_features=10):
        
        self.n_pca_components = n_pca_components
        self.n_ica_components = n_ica_components
        self.poly_degree = poly_degree
        self.n_clusters = n_clusters
        self.create_interactions = create_interactions
        self.create_polynomial = create_polynomial
        self.use_pca = use_pca
        self.use_ica = use_ica
        self.variance_threshold = variance_threshold
        self.n_stat_features = n_stat_features
        
        # 初始化转换器
        self.scaler = StandardScaler()
        self.pca = None
        self.ica = None
        self.poly = None
        self.kmeans = None
        
        # 记录特征信息
        self.original_feature_names = None
        self.engineered_feature_names = None
        self.feature_info = {}
    
    def fit(self, X, y=None):
        """拟合特征工程管道"""
        print("拟合特征工程管道...")
        
        # 保存原始特征名称
        if hasattr(X, 'columns'):
            self.original_feature_names = X.columns.tolist()
        else:
            self.original_feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # 1. 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. 移除低方差特征
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var_selected = self.variance_selector.fit_transform(X_scaled)
        
        # 3. PCA降维
        if self.use_pca and X_var_selected.shape[1] > 10:
            n_components = min(self.n_pca_components, X_var_selected.shape[1])
            if n_components < 1:
                self.pca = PCA(n_components=n_components, random_state=42)
            else:
                self.pca = PCA(n_components=int(n_components), random_state=42)
            self.pca.fit(X_var_selected)
        
        # 4. ICA降维
        if self.use_ica and X_var_selected.shape[1] > 5:
            self.ica = FastICA(n_components=min(self.n_ica_components, X_var_selected.shape[1]), 
                              random_state=42, max_iter=500)
            self.ica.fit(X_var_selected)
        
        # 5. 聚类特征
        if X_var_selected.shape[0] > self.n_clusters:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(X_var_selected)
        
        # 6. 多项式特征
        if self.create_polynomial and X_var_selected.shape[1] < 50:
            self.poly = PolynomialFeatures(degree=self.poly_degree, 
                                          include_bias=False, 
                                          interaction_only=False)
            # 只拟合部分特征以防止维度爆炸
            n_poly_features = min(10, X_var_selected.shape[1])
            self.poly.fit(X_var_selected[:, :n_poly_features])
        
        return self
    
    def transform(self, X):
        """应用特征工程转换"""
        print("应用特征工程转换...")
        
        # 存储所有生成的特征
        features_list = []
        feature_names_list = []
        
        # 1. 原始特征（标准化后）
        X_scaled = self.scaler.transform(X)
        features_list.append(X_scaled)
        feature_names_list.extend(self.original_feature_names)
        
        # 2. 移除低方差特征
        if hasattr(self, 'variance_selector'):
            X_var = self.variance_selector.transform(X_scaled)
            # 使用方差选择后的特征
            if X_var.shape[1] > 0:
                features_list.append(X_var)
                feature_names_list.extend([f'{self.original_feature_names[i]}_var' 
                                          for i in self.variance_selector.get_support(indices=True)])
        
        # 3. PCA特征
        if self.use_pca and self.pca is not None:
            X_pca = self.pca.transform(X_scaled)
            features_list.append(X_pca)
            feature_names_list.extend([f'PC_{i+1}' for i in range(X_pca.shape[1])])
        
        # 4. ICA特征
        if self.use_ica and self.ica is not None:
            X_ica = self.ica.transform(X_scaled)
            features_list.append(X_ica)
            feature_names_list.extend([f'ICA_{i+1}' for i in range(X_ica.shape[1])])
        
        # 5. 聚类特征
        if self.kmeans is not None:
            clusters = self.kmeans.predict(X_scaled).reshape(-1, 1)
            features_list.append(clusters)
            feature_names_list.extend(['Cluster'])
            
            # 聚类距离特征
            distances = self.kmeans.transform(X_scaled)
            features_list.append(distances)
            feature_names_list.extend([f'Dist_Cluster_{i}' for i in range(distances.shape[1])])
        
        # 6. 多项式特征
        if self.create_polynomial and self.poly is not None:
            n_poly_features = min(10, X_scaled.shape[1])
            X_poly = self.poly.transform(X_scaled[:, :n_poly_features])
            features_list.append(X_poly)
            # 生成多项式特征名称
            poly_names = []
            for powers in self.poly.powers_:
                name_parts = []
                for i, power in enumerate(powers):
                    if power > 0:
                        if power == 1:
                            name_parts.append(f"{self.original_feature_names[i]}")
                        else:
                            name_parts.append(f"{self.original_feature_names[i]}^{power}")
                poly_names.append('×'.join(name_parts))
            feature_names_list.extend(poly_names)
        
        # 7. 交互特征
        if self.create_interactions and X_scaled.shape[1] > 5:
            # 创建最重要的特征之间的交互
            n_interact = min(5, X_scaled.shape[1])
            interaction_features = []
            interaction_names = []
            
            for i in range(n_interact):
                for j in range(i+1, n_interact):
                    interaction = X_scaled[:, i] * X_scaled[:, j]
                    interaction_features.append(interaction)
                    interaction_names.append(f"{self.original_feature_names[i]}_x_{self.original_feature_names[j]}")
            
            if interaction_features:
                X_interact = np.column_stack(interaction_features)
                features_list.append(X_interact)
                feature_names_list.extend(interaction_names)
        
        # 8. 统计特征
        X_stats = self._create_statistical_features(X_scaled)
        if X_stats.shape[1] > 0:
            features_list.append(X_stats)
            feature_names_list.extend([f'Stat_{i}' for i in range(X_stats.shape[1])])
        
        # 合并所有特征
        X_engineered = np.hstack(features_list)
        self.engineered_feature_names = feature_names_list
        
        print(f"特征工程完成: {X.shape[1]} -> {X_engineered.shape[1]} 个特征")
        self.feature_info = {
            'original_features': X.shape[1],
            'engineered_features': X_engineered.shape[1],
            'methods_used': ['标准化', '方差选择', 'PCA', 'ICA', '聚类', '多项式', '交互', '统计']
        }
        
        return X_engineered
    
    def _create_statistical_features(self, X):
        """创建统计特征"""
        stats_features = []
        
        # 基本统计量
        stats_features.append(X.mean(axis=1).reshape(-1, 1))  # 均值
        stats_features.append(X.std(axis=1).reshape(-1, 1))   # 标准差
        stats_features.append(X.var(axis=1).reshape(-1, 1))   # 方差
        
        # 修正：使用scipy的stats模块正确计算偏度和峰度
        try:
            # 对于偏度
            skewness = np.apply_along_axis(stats.skew, 1, X)
            stats_features.append(skewness.reshape(-1, 1))
            
            # 对于峰度，使用scipy的kurtosis函数（注意：这里计算的是超额峰度）
            kurt = np.apply_along_axis(stats.kurtosis, 1, X)
            stats_features.append(kurt.reshape(-1, 1))
        except Exception as e:
            print(f"计算偏度/峰度时出错: {e}，使用零值替代")
            stats_features.append(np.zeros((X.shape[0], 1)))
            stats_features.append(np.zeros((X.shape[0], 1)))
        
        # 分位数
        for q in [0.25, 0.5, 0.75]:
            stats_features.append(np.percentile(X, q*100, axis=1).reshape(-1, 1))
        
        # 极差
        stats_features.append((X.max(axis=1) - X.min(axis=1)).reshape(-1, 1))
        
        return np.hstack(stats_features)
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.engineered_feature_names

class IntelligentFeatureSelector:
    """智能特征选择器"""
    
    def __init__(self, n_features='auto', selection_methods=None, 
                 n_iter_select=10, cv_folds=5):
        
        self.n_features = n_features
        self.selection_methods = selection_methods or ['lasso', 'rf', 'xgb', 'mutual_info', 'variance']
        self.n_iter_select = n_iter_select
        self.cv_folds = cv_folds
        
        # 结果存储
        self.best_method = None
        self.best_score = -np.inf
        self.best_features = None
        self.best_selector = None
        self.results = {}
        self.feature_importances = {}
        self.selected_feature_names = []
    
    def select_best_features(self, X, y, feature_names=None):
        """选择最佳特征子集"""
        print("\n" + "="*60)
        print("智能特征选择开始")
        print("="*60)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # 确定特征数量
        if self.n_features == 'auto':
            # 自动确定特征数量：特征数的平方根或20，取较小值
            n_features_auto = min(int(np.sqrt(X.shape[1])), 20, X.shape[1])
            n_features_auto = max(n_features_auto, 5)  # 至少5个特征
            self.n_features = n_features_auto
        print(f"目标特征数量: {self.n_features}")
        
        # 尝试不同的特征选择方法
        for method in self.selection_methods:
            print(f"尝试方法: {method}")
            
            try:
                if method == 'lasso':
                    score, features, selector = self._lasso_selection(X, y)
                elif method == 'rf':
                    score, features, selector = self._random_forest_selection(X, y)
                elif method == 'xgb':
                    score, features, selector = self._xgboost_selection(X, y)
                elif method == 'mutual_info':
                    score, features, selector = self._mutual_info_selection(X, y)
                elif method == 'variance':
                    score, features, selector = self._variance_selection(X)
                elif method == 'rfe':
                    score, features, selector = self._rfe_selection(X, y)
                elif method == 'pca':
                    score, features, selector = self._pca_selection(X, y)
                else:
                    continue
                
                # 存储结果
                self.results[method] = {
                    'score': score,
                    'n_features': len(features),
                    'features': features,
                    'selector': selector
                }
                
                print(f"  得分: {score:.4f}, 特征数: {len(features)}")
                
                # 更新最佳方法
                if score > self.best_score:
                    self.best_score = score
                    self.best_method = method
                    self.best_features = features
                    self.best_selector = selector
                    
            except Exception as e:
                print(f"  方法失败: {e}")
                continue
        
        # 如果没有方法成功，使用所有特征
        if self.best_method is None:
            print("所有特征选择方法失败，使用所有特征")
            self.best_features = np.arange(X.shape[1])
            self.best_score = 0.5  # 默认得分
            self.best_method = 'none'
        
        print(f"\n最佳特征选择方法: {self.best_method}")
        print(f"最佳得分: {self.best_score:.4f}")
        print(f"选择特征数: {len(self.best_features)}")
        
        # 获取选择的特征名称
        selected_feature_names = []
        if self.best_method == 'pca':
            # 对于PCA，使用组件的名称
            selected_feature_names = [f'PC_{i+1}' for i in range(len(self.best_features))]
        else:
            # 对于其他方法，使用原始特征名称
            selected_feature_names = []
            for idx in self.best_features:
                if idx < len(feature_names):
                    selected_feature_names.append(feature_names[idx])
                else:
                    selected_feature_names.append(f'Feature_{idx}')
        
        self.selected_feature_names = selected_feature_names
        
        return X[:, self.best_features], selected_feature_names
    
    def _lasso_selection(self, X, y):
        """Lasso特征选择"""
        # 对于分类问题，使用LogisticRegression的L1正则化
        if len(np.unique(y)) <= 2:
            # 二分类问题
            lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
            lasso.fit(X, y)
            selected_features = np.where(lasso.coef_[0] != 0)[0]
        else:
            # 回归问题或使用LassoCV
            lasso_cv = LassoCV(cv=min(5, X.shape[0]//2), random_state=42, max_iter=5000)
            lasso_cv.fit(X, y)
            selected_features = np.where(lasso_cv.coef_ != 0)[0]
        
        if len(selected_features) == 0:
            # 如果没有特征被选中，选择系数绝对值最大的特征
            n_features = min(self.n_features, X.shape[1])
            if len(np.unique(y)) <= 2:
                selected_features = np.argsort(np.abs(lasso.coef_[0]))[-n_features:]
            else:
                selected_features = np.argsort(np.abs(lasso_cv.coef_))[-n_features:]
        
        # 评估特征子集
        score = self._evaluate_feature_subset(X[:, selected_features], y)
        
        return score, selected_features, lasso_cv if len(np.unique(y)) > 2 else lasso
    
    def _random_forest_selection(self, X, y):
        """随机森林特征选择"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        
        # 选择最重要的特征
        n_features = min(self.n_features, X.shape[1])
        selected_features = np.argsort(importances)[-n_features:]
        
        # 存储特征重要性
        for idx, imp in zip(selected_features, importances[selected_features]):
            if idx not in self.feature_importances:
                self.feature_importances[idx] = []
            self.feature_importances[idx].append(('rf', imp))
        
        # 评估特征子集
        score = self._evaluate_feature_subset(X[:, selected_features], y)
        
        return score, selected_features, rf
    
    def _xgboost_selection(self, X, y):
        """XGBoost特征选择"""
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X, y)
            
            # 获取特征重要性
            importances = xgb_model.feature_importances_
            
            # 选择最重要的特征
            n_features = min(self.n_features, X.shape[1])
            selected_features = np.argsort(importances)[-n_features:]
            
            # 存储特征重要性
            for idx, imp in zip(selected_features, importances[selected_features]):
                if idx not in self.feature_importances:
                    self.feature_importances[idx] = []
                self.feature_importances[idx].append(('xgb', imp))
            
            # 评估特征子集
            score = self._evaluate_feature_subset(X[:, selected_features], y)
            
            return score, selected_features, xgb_model
        except ImportError:
            print("XGBoost未安装，使用梯度提升替代")
            # 如果XGBoost未安装，使用梯度提升
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X, y)
            
            importances = gb.feature_importances_
            n_features = min(self.n_features, X.shape[1])
            selected_features = np.argsort(importances)[-n_features:]
            
            for idx, imp in zip(selected_features, importances[selected_features]):
                if idx not in self.feature_importances:
                    self.feature_importances[idx] = []
                self.feature_importances[idx].append(('gb', imp))
            
            score = self._evaluate_feature_subset(X[:, selected_features], y)
            
            return score, selected_features, gb
    
    def _mutual_info_selection(self, X, y):
        """互信息特征选择"""
        # 计算互信息得分
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 选择得分最高的特征
        n_features = min(self.n_features, X.shape[1])
        selected_features = np.argsort(mi_scores)[-n_features:]
        
        # 存储特征重要性
        for idx, score in zip(selected_features, mi_scores[selected_features]):
            if idx not in self.feature_importances:
                self.feature_importances[idx] = []
            self.feature_importances[idx].append(('mi', score))
        
        # 评估特征子集
        score = self._evaluate_feature_subset(X[:, selected_features], y)
        
        return score, selected_features, None
    
    def _variance_selection(self, X):
        """方差特征选择"""
        selector = VarianceThreshold(threshold=np.var(X, axis=0).mean() * 0.1)
        selector.fit(X)
        selected_features = selector.get_support(indices=True)
        
        if len(selected_features) == 0:
            # 如果没有特征被选中，选择方差最大的特征
            variances = np.var(X, axis=0)
            n_features = min(self.n_features, X.shape[1])
            selected_features = np.argsort(variances)[-n_features:]
        
        # 由于没有y，返回默认得分
        score = 0.5
        
        return score, selected_features, selector
    
    def _rfe_selection(self, X, y):
        """递归特征消除"""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=self.n_features)
        rfe.fit(X, y)
        
        selected_features = np.where(rfe.support_)[0]
        
        # 评估特征子集
        score = self._evaluate_feature_subset(X[:, selected_features], y)
        
        return score, selected_features, rfe
    
    def _pca_selection(self, X, y):
        """PCA特征选择"""
        n_components = min(self.n_features, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # 使用PCA转换后的特征进行评估
        score = self._evaluate_feature_subset(X_pca, y)
        
        # PCA没有特征索引，返回虚拟索引
        selected_features = np.arange(n_components)
        
        return score, selected_features, pca
    
    def _evaluate_feature_subset(self, X_subset, y):
        """评估特征子集"""
        # 使用简单的随机森林进行快速评估
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        # 使用交叉验证评估
        cv_scores = cross_val_score(rf, X_subset, y, cv=min(3, X_subset.shape[0]//2), 
                                   scoring='accuracy', n_jobs=-1)
        
        return cv_scores.mean()
    
    def transform(self, X):
        """应用最佳特征选择器"""
        if self.best_selector is not None:
            if self.best_method == 'pca':
                return self.best_selector.transform(X)
            elif hasattr(self.best_selector, 'transform'):
                return self.best_selector.transform(X)
        
        # 如果没有选择器或方法不支持transform，直接选择特征
        if self.best_features is not None:
            return X[:, self.best_features]
        
        return X
    
    def plot_selection_results(self, feature_names):
        """可视化特征选择结果"""
        if not self.results:
            print("没有特征选择结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flat
        
        # 1. 方法性能比较
        if len(self.results) > 1:
            methods = list(self.results.keys())
            scores = [self.results[m]['score'] for m in methods]
            n_features = [self.results[m]['n_features'] for m in methods]
            
            ax = axes[0]
            bars = ax.bar(methods, scores, color=['skyblue', 'lightgreen', 'lightcoral', 
                                                 'gold', 'violet', 'orange'][:len(methods)])
            
            # 标记最佳方法
            if self.best_method in methods:
                best_idx = methods.index(self.best_method)
                bars[best_idx].set_color('red')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
            
            ax.set_xlabel('特征选择方法')
            ax.set_ylabel('交叉验证得分')
            ax.set_title('特征选择方法性能比较')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for i, (bar, score, n_feat) in enumerate(zip(bars, scores, n_features)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}\n({n_feat} feat)',
                       ha='center', va='bottom', fontsize=9)
        
        # 2. 特征重要性（如果可用）
        if self.feature_importances:
            # 计算每个特征的平均重要性
            feature_scores = {}
            for feat_idx, scores_list in self.feature_importances.items():
                avg_score = np.mean([score for _, score in scores_list])
                feature_scores[feat_idx] = avg_score
            
            # 选择最重要的20个特征
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            top_n = min(20, len(sorted_features))
            top_features = sorted_features[:top_n]
            
            feat_indices = [f[0] for f in top_features]
            feat_scores = [f[1] for f in top_features]
            
            # 获取特征名称
            feat_names = []
            for idx in feat_indices:
                if idx < len(feature_names):
                    feat_names.append(feature_names[idx])
                else:
                    feat_names.append(f'Feature_{idx}')
            
            ax = axes[1]
            y_pos = np.arange(len(feat_names))
            ax.barh(y_pos, feat_scores, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names)
            ax.set_xlabel('平均特征重要性')
            ax.set_title(f'Top {top_n} 最重要的特征')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 3. 特征数量与得分关系
        if len(self.results) > 2:
            ax = axes[2]
            for method, result in self.results.items():
                if 'n_features' in result and 'score' in result:
                    ax.scatter(result['n_features'], result['score'], 
                              s=100, alpha=0.7, label=method)
            
            ax.set_xlabel('特征数量')
            ax.set_ylabel('得分')
            ax.set_title('特征数量 vs 模型性能')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. 最佳方法详细信息
        ax = axes[3]
        if self.best_method:
            result = self.results.get(self.best_method, {})
            info_text = f"最佳方法: {self.best_method}\n"
            info_text += f"得分: {self.best_score:.4f}\n"
            info_text += f"特征数: {len(self.best_features)}\n"
            
            # 显示前10个特征名称
            if self.best_features is not None and feature_names:
                top_features = self.best_features[:10]
                feature_list = []
                for i in top_features:
                    if i < len(feature_names):
                        feature_list.append(feature_names[i])
                    else:
                        feature_list.append(f'Feature_{i}')
                
                info_text += f"\n前10个特征:\n"
                for i, feat in enumerate(feature_list[:10]):
                    info_text += f"  {i+1}. {feat}\n"
            
            ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='center')
            ax.set_title('最佳特征选择方法详情')
            ax.axis('off')
        
        # 5. 特征选择方法统计
        ax = axes[4]
        if self.results:
            methods = list(self.results.keys())
            n_features_list = [self.results[m]['n_features'] for m in methods]
            
            bars = ax.bar(methods, n_features_list, color='lightgreen')
            ax.set_xlabel('特征选择方法')
            ax.set_ylabel('选择的特征数量')
            ax.set_title('各方法选择的特征数量')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. 特征选择进度（占位）
        ax = axes[5]
        ax.text(0.5, 0.5, '特征选择完成', transform=ax.transAxes,
               fontsize=14, ha='center', va='center', fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/feature_selection_analysis.png', dpi=300, bbox_inches='tight')
        print("特征选择分析图已保存为 'results/feature_selection_analysis.png'")
        plt.show()

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1):
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_models = {}
        self.optimization_history = {}
    
    def get_model_configs(self):
        """获取模型配置和参数网格"""
        configs = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'param_distributions': {
                    'n_estimators': randint(50, 500),
                    'max_depth': [None, 10, 20, 30, 50],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                },
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'learning_rate': loguniform(0.001, 0.3),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'subsample': uniform(0.5, 0.5)
                },
                'param_grid': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'param_distributions': {
                    'C': loguniform(0.1, 100),
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),
                    'degree': randint(2, 5)
                },
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_distributions': {
                    'C': loguniform(0.001, 100),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                },
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'param_distributions': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 50, 50)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': loguniform(0.0001, 0.1),
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': loguniform(0.001, 0.1)
                },
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        
        # 尝试添加XGBoost配置
        try:
            import xgboost as xgb
            configs['XGBoost'] = {
                'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'param_distributions': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': loguniform(0.001, 0.3),
                    'subsample': uniform(0.5, 0.5),
                    'colsample_bytree': uniform(0.5, 0.5),
                    'gamma': uniform(0, 5)
                },
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
            }
        except ImportError:
            print("XGBoost未安装，跳过XGBoost模型")
        
        return configs
    
    def optimize_model(self, model_name, X_train, y_train, X_val=None, y_val=None):
        """优化单个模型"""
        print(f"优化模型: {model_name}")
        
        configs = self.get_model_configs()
        if model_name not in configs:
            print(f"模型 '{model_name}' 配置不存在")
            return None
        
        config = configs[model_name]
        model = config['model']
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 使用随机搜索进行初始优化
            random_search = RandomizedSearchCV(
                model,
                config['param_distributions'],
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=42,
                verbose=0
            )
            
            random_search.fit(X_train, y_train)
            
            # 在随机搜索的基础上进行网格搜索微调
            best_params = random_search.best_params_
            
            # 创建更精细的参数网格进行微调
            refined_grid = {}
            for param, value in best_params.items():
                if param in config['param_grid']:
                    # 如果参数在网格中，使用网格值
                    refined_grid[param] = config['param_grid'][param]
                elif isinstance(value, (int, np.integer)):
                    # 对于整数参数，创建周围的值
                    refined_grid[param] = [max(1, value-2), value, value+2]
                elif isinstance(value, float):
                    # 对于浮点数参数，创建对数范围
                    refined_grid[param] = [value/2, value, value*2]
                else:
                    refined_grid[param] = [value]
            
            # 网格搜索微调
            grid_search = GridSearchCV(
                clone(model),
                refined_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # 计算总时间
            total_time = time.time() - start_time
            
            # 评估模型
            best_model = grid_search.best_estimator_
            train_score = grid_search.best_score_
            
            # 如果有验证集，评估验证集性能
            val_score = None
            if X_val is not None and y_val is not None:
                y_val_pred = best_model.predict(X_val)
                val_score = accuracy_score(y_val, y_val_pred)
            
            print(f"  优化完成 - 时间: {total_time:.1f}秒")
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  训练得分: {train_score:.4f}")
            if val_score is not None:
                print(f"  验证得分: {val_score:.4f}")
            
            # 存储优化结果
            self.best_models[model_name] = {
                'model': best_model,
                'params': grid_search.best_params_,
                'train_score': train_score,
                'val_score': val_score,
                'optimization_time': total_time,
                'cv_results': grid_search.cv_results_
            }
            
            self.optimization_history[model_name] = {
                'random_search_best': random_search.best_score_,
                'grid_search_best': train_score,
                'improvement': train_score - random_search.best_score_
            }
            
            return best_model
            
        except Exception as e:
            print(f"  优化失败: {e}")
            return None
    
    def optimize_all_models(self, X_train, y_train, models_to_optimize=None):
        """优化所有模型"""
        print("\n" + "="*60)
        print("开始模型超参数优化")
        print("="*60)
        
        # 分割验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"训练集: {X_train_final.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        
        configs = self.get_model_configs()
        if models_to_optimize is None:
            models_to_optimize = list(configs.keys())
        
        for model_name in models_to_optimize:
            if model_name in configs:
                self.optimize_model(model_name, X_train_final, y_train_final, X_val, y_val)
        
        # 找到最佳模型
        best_model_name = None
        best_val_score = -np.inf
        
        for model_name, result in self.best_models.items():
            if result['val_score'] is not None and result['val_score'] > best_val_score:
                best_val_score = result['val_score']
                best_model_name = model_name
        
        if best_model_name:
            print(f"\n最佳模型: {best_model_name} (验证得分: {best_val_score:.4f})")
            return self.best_models[best_model_name]['model'], best_model_name
        else:
            print("未找到最佳模型")
            return None, None
    
    def plot_optimization_results(self):
        """可视化优化结果"""
        if not self.best_models:
            print("没有优化结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flat
        
        # 1. 模型性能比较
        model_names = list(self.best_models.keys())
        train_scores = [self.best_models[m]['train_score'] for m in model_names]
        val_scores = [self.best_models[m]['val_score'] or 0 for m in model_names]
        opt_times = [self.best_models[m]['optimization_time'] for m in model_names]
        
        ax = axes[0]
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='训练得分', color='skyblue')
        bars2 = ax.bar(x + width/2, val_scores, width, label='验证得分', color='lightcoral')
        
        ax.set_xlabel('模型')
        ax.set_ylabel('得分')
        ax.set_title('模型性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标记最佳模型
        if val_scores:
            best_idx = np.argmax(val_scores)
            bars2[best_idx].set_color('red')
            bars2[best_idx].set_edgecolor('black')
            bars2[best_idx].set_linewidth(2)
        
        # 2. 优化时间比较
        ax = axes[1]
        bars = ax.bar(model_names, opt_times, color='lightgreen')
        ax.set_xlabel('模型')
        ax.set_ylabel('优化时间 (秒)')
        ax.set_title('模型优化时间比较')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加时间
        for bar, time_val in zip(bars, opt_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 3. 优化提升效果
        ax = axes[2]
        if self.optimization_history:
            improvements = []
            for model_name in model_names:
                if model_name in self.optimization_history:
                    improvements.append(self.optimization_history[model_name]['improvement'])
                else:
                    improvements.append(0)
            
            bars = ax.bar(model_names, improvements, color='gold')
            ax.set_xlabel('模型')
            ax.set_ylabel('提升幅度')
            ax.set_title('参数优化带来的性能提升')
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{imp:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 最佳模型参数详情
        ax = axes[3]
        best_model_name = None
        best_val_score = -np.inf
        
        for model_name, result in self.best_models.items():
            if result['val_score'] is not None and result['val_score'] > best_val_score:
                best_val_score = result['val_score']
                best_model_name = model_name
        
        if best_model_name:
            best_result = self.best_models[best_model_name]
            info_text = f"最佳模型: {best_model_name}\n"
            info_text += f"验证得分: {best_val_score:.4f}\n"
            info_text += f"训练得分: {best_result['train_score']:.4f}\n"
            info_text += f"优化时间: {best_result['optimization_time']:.1f}秒\n\n"
            info_text += "最佳参数:\n"
            
            for param, value in best_result['params'].items():
                info_text += f"  {param}: {value}\n"
            
            ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='center', family='monospace')
            ax.set_title('最佳模型详情')
            ax.axis('off')
        
        # 5. 性能与时间关系
        ax = axes[4]
        scatter = ax.scatter(opt_times, val_scores, s=100, alpha=0.7, c='steelblue')
        
        # 标记最佳模型
        if best_model_name and best_model_name in model_names:
            best_idx = model_names.index(best_model_name)
            ax.scatter(opt_times[best_idx], val_scores[best_idx], 
                      s=200, c='red', marker='*', edgecolors='black')
        
        ax.set_xlabel('优化时间 (秒)')
        ax.set_ylabel('验证得分')
        ax.set_title('优化时间 vs 模型性能')
        ax.grid(True, alpha=0.3)
        
        # 添加模型标签
        for i, name in enumerate(model_names):
            ax.annotate(name, (opt_times[i], val_scores[i]), 
                       fontsize=8, alpha=0.7)
        
        # 6. 优化方法对比
        ax = axes[5]
        if self.optimization_history:
            random_scores = []
            grid_scores = []
            
            for model_name in model_names:
                if model_name in self.optimization_history:
                    random_scores.append(self.optimization_history[model_name]['random_search_best'])
                    grid_scores.append(self.optimization_history[model_name]['grid_search_best'])
                else:
                    random_scores.append(0)
                    grid_scores.append(0)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, random_scores, width, label='随机搜索', color='lightblue')
            ax.bar(x + width/2, grid_scores, width, label='网格搜索', color='lightcoral')
            
            ax.set_xlabel('模型')
            ax.set_ylabel('得分')
            ax.set_title('优化方法对比')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/hyperparameter_optimization_results.png', dpi=300, bbox_inches='tight')
        print("超参数优化结果图已保存为 'results/hyperparameter_optimization_results.png'")
        plt.show()

class EnsembleModelBuilder:
    """集成模型构建器"""
    
    def __init__(self, base_models=None):
        self.base_models = base_models or {}
        self.ensemble_model = None
        self.ensemble_score = 0
    
    def build_voting_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """构建投票集成模型"""
        print("构建投票集成模型...")
        
        if len(self.base_models) < 2:
            print("至少需要2个模型来构建集成模型")
            return None
        
        # 创建投票分类器
        estimators = [(name, model) for name, model in self.base_models.items()]
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 使用软投票（概率平均）
            n_jobs=-1
        )
        
        # 训练集成模型
        voting_clf.fit(X_train, y_train)
        
        # 评估集成模型
        train_score = voting_clf.score(X_train, y_train)
        
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = voting_clf.score(X_val, y_val)
        
        print(f"集成模型训练完成")
        print(f"训练得分: {train_score:.4f}")
        if val_score is not None:
            print(f"验证得分: {val_score:.4f}")
        
        self.ensemble_model = voting_clf
        self.ensemble_score = val_score or train_score
        
        return voting_clf
    
    def build_stacking_ensemble(self, X_train, y_train, X_val=None, y_val=None, 
                               meta_model=None):
        """构建堆叠集成模型"""
        print("构建堆叠集成模型...")
        
        if len(self.base_models) < 2:
            print("至少需要2个模型来构建堆叠集成")
            return None
        
        # 使用逻辑回归作为元模型
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # 创建堆叠分类器
        estimators = [(name, model) for name, model in self.base_models.items()]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # 训练堆叠模型
        stacking_clf.fit(X_train, y_train)
        
        # 评估堆叠模型
        train_score = stacking_clf.score(X_train, y_train)
        
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = stacking_clf.score(X_val, y_val)
        
        print(f"堆叠模型训练完成")
        print(f"训练得分: {train_score:.4f}")
        if val_score is not None:
            print(f"验证得分: {val_score:.4f}")
        
        return stacking_clf
    
    def get_best_ensemble(self, X_train, y_train, X_val, y_val):
        """获取最佳集成模型"""
        print("\n" + "="*60)
        print("构建集成模型")
        print("="*60)
        
        # 构建投票集成
        voting_clf = self.build_voting_ensemble(X_train, y_train, X_val, y_val)
        
        # 构建堆叠集成
        stacking_clf = self.build_stacking_ensemble(X_train, y_train, X_val, y_val)
        
        # 比较集成模型性能
        ensemble_scores = {}
        
        if voting_clf is not None:
            voting_score = voting_clf.score(X_val, y_val)
            ensemble_scores['VotingEnsemble'] = {'model': voting_clf, 'score': voting_score}
        
        if stacking_clf is not None:
            stacking_score = stacking_clf.score(X_val, y_val)
            ensemble_scores['StackingEnsemble'] = {'model': stacking_clf, 'score': stacking_score}
        
        # 选择最佳集成模型
        if ensemble_scores:
            best_ensemble_name = max(ensemble_scores.items(), key=lambda x: x[1]['score'])[0]
            best_ensemble = ensemble_scores[best_ensemble_name]['model']
            best_score = ensemble_scores[best_ensemble_name]['score']
            
            print(f"\n最佳集成模型: {best_ensemble_name} (得分: {best_score:.4f})")
            
            return best_ensemble, best_ensemble_name, ensemble_scores
        
        return None, None, {}

class AutoMLPipeline:
    """自动化机器学习管道"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.feature_engineering = FeatureEngineeringPipeline(
            n_pca_components=0.95,
            n_ica_components=10,
            poly_degree=2,
            n_clusters=5,
            create_interactions=True,
            create_polynomial=True,
            use_pca=True,
            use_ica=True,
            variance_threshold=0.01
        )
        
        self.feature_selector = IntelligentFeatureSelector(
            n_features='auto',
            selection_methods=['lasso', 'rf', 'xgb', 'mutual_info'],
            n_iter_select=10,
            cv_folds=5
        )
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            n_iter=15,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        self.ensemble_builder = EnsembleModelBuilder()
        
        # 存储结果
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        self.execution_time = 0
        self.selected_feature_names = []
        
    def run(self):
        """运行自动化机器学习管道"""
        print("="*80)
        print("自动化机器学习管道启动")
        print("="*80)
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        start_time = time.time()
        
        try:
            # 1. 加载数据
            print("\n" + "="*60)
            print("步骤1: 加载数据")
            print("="*60)
            
            X_train, y_train, X_test, y_test, feature_names = self.data_loader.load_data()
            
            if X_train is None:
                print("数据加载失败，终止流程")
                return
            
            # 2. 特征工程
            print("\n" + "="*60)
            print("步骤2: 自动化特征工程")
            print("="*60)
            
            # 拟合特征工程管道
            self.feature_engineering.fit(X_train, y_train)
            
            # 转换训练和测试数据
            X_train_engineered = self.feature_engineering.transform(X_train)
            X_test_engineered = self.feature_engineering.transform(X_test)
            
            engineered_feature_names = self.feature_engineering.get_feature_names()
            
            print(f"特征工程完成:")
            print(f"  原始特征: {X_train.shape[1]} 个")
            print(f"  工程后特征: {X_train_engineered.shape[1]} 个")
            
            # 3. 特征选择
            print("\n" + "="*60)
            print("步骤3: 智能特征选择")
            print("="*60)
            
            X_train_selected, selected_feature_names = self.feature_selector.select_best_features(
                X_train_engineered, y_train, engineered_feature_names
            )
            
            X_test_selected = self.feature_selector.transform(X_test_engineered)
            
            self.selected_feature_names = selected_feature_names
            
            print(f"特征选择完成:")
            print(f"  选择方法: {self.feature_selector.best_method}")
            print(f"  特征数量: {X_train_selected.shape[1]} 个")
            
            # 可视化特征选择结果
            self.feature_selector.plot_selection_results(selected_feature_names)
            
            # 4. 模型训练和超参数优化
            print("\n" + "="*60)
            print("步骤4: 模型超参数优化")
            print("="*60)
            
            # 优化所有模型
            best_single_model, best_single_model_name = self.hyperparameter_optimizer.optimize_all_models(
                X_train_selected, y_train
            )
            
            # 可视化优化结果
            self.hyperparameter_optimizer.plot_optimization_results()
            
            # 5. 构建集成模型
            print("\n" + "="*60)
            print("步骤5: 构建集成模型")
            print("="*60)
            
            # 准备基础模型
            base_models = {}
            for model_name, result in self.hyperparameter_optimizer.best_models.items():
                base_models[model_name] = result['model']
            
            self.ensemble_builder.base_models = base_models
            
            # 分割验证集用于集成模型评估
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # 构建集成模型
            best_ensemble, best_ensemble_name, ensemble_results = self.ensemble_builder.get_best_ensemble(
                X_train_final, y_train_final, X_val, y_val
            )
            
            # 6. 模型评估和选择
            print("\n" + "="*60)
            print("步骤6: 最终模型评估")
            print("="*60)
            
            # 评估所有模型在测试集上的性能
            all_models = {}
            
            # 添加单个模型
            for model_name, result in self.hyperparameter_optimizer.best_models.items():
                model = result['model']
                y_pred = model.predict(X_test_selected)
                test_acc = accuracy_score(y_test, y_pred)
                
                # 获取详细分类报告
                report = classification_report(y_test, y_pred, output_dict=True)
                
                all_models[model_name] = {
                    'model': model,
                    'test_accuracy': test_acc,
                    'type': 'single',
                    'classification_report': report
                }
            
            # 添加集成模型
            if best_ensemble is not None:
                y_pred_ensemble = best_ensemble.predict(X_test_selected)
                ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
                
                # 获取详细分类报告
                report = classification_report(y_test, y_pred_ensemble, output_dict=True)
                
                all_models[best_ensemble_name] = {
                    'model': best_ensemble,
                    'test_accuracy': ensemble_acc,
                    'type': 'ensemble',
                    'classification_report': report
                }
            
            # 选择最佳模型
            if all_models:
                self.best_model_name = max(all_models.items(), key=lambda x: x[1]['test_accuracy'])[0]
                self.best_model = all_models[self.best_model_name]['model']
                self.best_score = all_models[self.best_model_name]['test_accuracy']
                
                print(f"\n最佳模型: {self.best_model_name}")
                print(f"测试集准确率: {self.best_score:.4f}")
            else:
                print("没有可用的模型进行评估")
                return
            
            # 7. 模型评估可视化
            print("\n" + "="*60)
            print("步骤7: 模型评估可视化")
            print("="*60)
            
            self._visualize_final_results(all_models, X_test_selected, y_test)
            
            # 8. 保存模型和结果
            print("\n" + "="*60)
            print("步骤8: 保存模型和结果")
            print("="*60)
            
            self._save_models_and_results(all_models)
            
            # 9. 新数据预测（可选）
            print("\n" + "="*60)
            print("步骤9: 新数据预测")
            print("="*60)
            
            self._predict_new_data()
            
            # 计算总执行时间
            self.execution_time = time.time() - start_time
            
            # 10. 生成最终报告
            print("\n" + "="*80)
            print("自动化机器学习管道完成!")
            print("="*80)
            
            self._generate_final_report(X_train.shape, X_train_selected.shape)
            
        except Exception as e:
            print(f"管道执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_final_results(self, all_models, X_test, y_test):
        """可视化最终结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flat
        
        # 1. 所有模型性能比较
        model_names = list(all_models.keys())
        test_accuracies = [all_models[name]['test_accuracy'] for name in model_names]
        model_types = [all_models[name]['type'] for name in model_names]
        
        ax = axes[0]
        colors = ['skyblue' if t == 'single' else 'lightcoral' for t in model_types]
        bars = ax.bar(model_names, test_accuracies, color=colors)
        
        # 标记最佳模型
        best_idx = test_accuracies.index(max(test_accuracies))
        bars[best_idx].set_color('red')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('测试集准确率')
        ax.set_title('所有模型性能比较')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, acc in zip(bars, test_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='单个模型'),
            Patch(facecolor='lightcoral', label='集成模型'),
            Patch(facecolor='red', label='最佳模型')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 2. 最佳模型混淆矩阵
        ax = axes[1]
        if self.best_model is not None:
            y_pred = self.best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # 使用seaborn绘制热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=sorted(np.unique(y_test)),
                       yticklabels=sorted(np.unique(y_test)))
            
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title(f'最佳模型混淆矩阵 ({self.best_model_name})')
        
        # 3. 分类报告热图
        ax = axes[2]
        if self.best_model is not None:
            y_pred = self.best_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # 提取指标
            metrics = ['precision', 'recall', 'f1-score']
            classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
            
            # 创建热图数据
            heatmap_data = []
            for cls in classes:
                row = [report[cls][metric] for metric in metrics]
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data)
            
            # 绘制热图
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # 设置坐标轴
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_xticklabels(metrics)
            ax.set_yticks(np.arange(len(classes)))
            ax.set_yticklabels([f'类别 {c}' for c in classes])
            
            # 添加数值
            for i in range(len(classes)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                  ha="center", va="center", color="black")
            
            ax.set_title('最佳模型分类报告')
        
        # 4. 特征数量变化
        ax = axes[3]
        stages = ['原始数据', '特征工程后', '特征选择后']
        feature_counts = [
            self.feature_engineering.feature_info.get('original_features', 0),
            self.feature_engineering.feature_info.get('engineered_features', 0),
            len(self.feature_selector.best_features) if self.feature_selector.best_features is not None else 0
        ]
        
        bars = ax.bar(stages, feature_counts, color=['lightblue', 'lightgreen', 'gold'])
        ax.set_xlabel('处理阶段')
        ax.set_ylabel('特征数量')
        ax.set_title('特征数量变化')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(count), ha='center', va='bottom', fontsize=10)
        
        # 5. 最佳模型特征重要性（如果可用）
        ax = axes[4]
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            # 选择最重要的20个特征
            n_features = min(20, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            # 获取特征名称
            if self.selected_feature_names:
                feature_names = self.selected_feature_names
                names = []
                for i in indices:
                    if i < len(feature_names):
                        names.append(feature_names[i])
                    else:
                        names.append(f'Feature_{i}')
            else:
                names = [f'Feature_{i}' for i in indices]
            
            ax.barh(range(n_features), importances[indices])
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('特征重要性')
            ax.set_title(f'最佳模型特征重要性 ({self.best_model_name})')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 6. 管道总结
        ax = axes[5]
        summary_text = f"最佳模型: {self.best_model_name}\n"
        summary_text += f"测试准确率: {self.best_score:.4f}\n\n"
        summary_text += f"总执行时间: {self.execution_time:.1f}秒\n\n"
        summary_text += f"特征工程方法:\n"
        for method in self.feature_engineering.feature_info.get('methods_used', []):
            summary_text += f"  • {method}\n"
        summary_text += f"\n特征选择方法: {self.feature_selector.best_method}\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center')
        ax.set_title('管道总结报告')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/final_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("最终模型评估图已保存为 'results/final_model_evaluation.png'")
        plt.show()
    
    def _save_models_and_results(self, all_models):
        """保存模型和结果"""
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 保存最佳模型
            best_model_path = f'models/best_model_{timestamp}.pkl'
            joblib.dump(self.best_model, best_model_path)
            print(f"最佳模型已保存: {best_model_path}")
            
            # 2. 保存所有模型
            all_models_path = f'models/all_models_{timestamp}.pkl'
            joblib.dump(all_models, all_models_path)
            print(f"所有模型已保存: {all_models_path}")
            
            # 3. 保存特征工程管道
            feature_engineering_path = f'models/feature_engineering_{timestamp}.pkl'
            joblib.dump(self.feature_engineering, feature_engineering_path)
            print(f"特征工程管道已保存: {feature_engineering_path}")
            
            # 4. 保存特征选择器
            feature_selector_path = f'models/feature_selector_{timestamp}.pkl'
            joblib.dump(self.feature_selector, feature_selector_path)
            print(f"特征选择器已保存: {feature_selector_path}")
            
            # 5. 保存超参数优化器
            optimizer_path = f'models/hyperparameter_optimizer_{timestamp}.pkl'
            joblib.dump(self.hyperparameter_optimizer, optimizer_path)
            print(f"超参数优化器已保存: {optimizer_path}")
            
            # 6. 保存详细结果到JSON
            pipeline_config = {
                'timestamp': timestamp,
                'best_model_name': self.best_model_name,
                'best_score': float(self.best_score),
                'execution_time': float(self.execution_time),
                'feature_engineering_info': self.feature_engineering.feature_info,
                'feature_selector_info': {
                    'best_method': self.feature_selector.best_method,
                    'best_score': float(self.feature_selector.best_score),
                    'n_selected_features': len(self.feature_selector.best_features) 
                    if self.feature_selector.best_features is not None else 0
                },
                'model_performance': {}
            }
            
            # 添加模型性能信息
            for name, info in all_models.items():
                pipeline_config['model_performance'][name] = {
                    'type': info['type'],
                    'test_accuracy': float(info['test_accuracy']),
                    'classification_report': info['classification_report']
                }
            
            config_path = f'results/pipeline_results_{timestamp}.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_config, f, ensure_ascii=False, indent=2)
            print(f"管道配置已保存: {config_path}")
            
            # 7. 保存为CSV格式的结果总结
            results_data = []
            for name, info in all_models.items():
                results_data.append({
                    'Model': name,
                    'Type': info['type'],
                    'Test_Accuracy': info['test_accuracy'],
                    'Precision_Avg': info['classification_report']['macro avg']['precision'],
                    'Recall_Avg': info['classification_report']['macro avg']['recall'],
                    'F1_Score_Avg': info['classification_report']['macro avg']['f1-score']
                })
            
            results_df = pd.DataFrame(results_data).sort_values('Test_Accuracy', ascending=False)
            csv_path = f'results/model_results_summary_{timestamp}.csv'
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"结果总结已保存: {csv_path}")
            
            # 8. 保存特征名称
            if self.selected_feature_names:
                features_df = pd.DataFrame({
                    'Index': range(len(self.selected_feature_names)),
                    'Feature_Name': self.selected_feature_names
                })
                features_path = f'results/selected_features_{timestamp}.csv'
                features_df.to_csv(features_path, index=False, encoding='utf-8-sig')
                print(f"特征名称已保存: {features_path}")
            
        except Exception as e:
            print(f"保存失败: {e}")
    
    def _predict_new_data(self):
        """对新数据进行预测"""
        try:
            # 加载新数据
            X_new, new_feature_names = self.data_loader.load_new_data()
            
            if X_new is None:
                return
            
            print(f"新数据: {X_new.shape[0]} 个样本")
            
            # 应用特征工程
            X_new_engineered = self.feature_engineering.transform(X_new)
            
            # 应用特征选择
            X_new_selected = self.feature_selector.transform(X_new_engineered)
            
            # 使用最佳模型进行预测
            predictions = self.best_model.predict(X_new_selected)
            
            # 如果有预测概率
            probabilities = None
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_new_selected)
            
            # 创建时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存预测结果
            results_df = pd.DataFrame({
                'Sample_ID': range(1, len(predictions) + 1),
                'Predicted_Class': predictions
            })
            
            # 添加概率列
            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    results_df[f'Class_{i}_Probability'] = probabilities[:, i]
            
            # 保存到Excel
            excel_path = f'results/new_data_predictions_{timestamp}.xlsx'
            results_df.to_excel(excel_path, index=False)
            print(f"新数据预测结果已保存: {excel_path}")
            
            # 保存到CSV
            csv_path = f'results/new_data_predictions_{timestamp}.csv'
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"新数据预测结果已保存: {csv_path}")
            
            # 显示预测统计
            print(f"\n预测结果统计:")
            print(f"总样本数: {len(predictions)}")
            
            unique_classes, class_counts = np.unique(predictions, return_counts=True)
            stats_data = []
            for cls, count in zip(unique_classes, class_counts):
                percentage = count / len(predictions) * 100
                stats_data.append({
                    'Class': cls,
                    'Count': count,
                    'Percentage': percentage
                })
                print(f"类别 {cls}: {count} 个样本 ({percentage:.1f}%)")
            
            # 保存统计信息
            stats_df = pd.DataFrame(stats_data)
            stats_path = f'results/prediction_statistics_{timestamp}.csv'
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            print(f"预测统计信息已保存: {stats_path}")
                
        except Exception as e:
            print(f"新数据预测失败: {e}")
    
    def _generate_final_report(self, original_shape, selected_shape):
        """生成最终报告"""
        print("\n" + "="*80)
        print("自动化机器学习管道最终报告")
        print("="*80)
        
        print(f"\n数据统计:")
        print(f"原始数据维度: {original_shape[0]} × {original_shape[1]}")
        print(f"处理后数据维度: {selected_shape[0]} × {selected_shape[1]}")
        if original_shape[1] > 0:
            reduction_percent = (1 - selected_shape[1]/original_shape[1])*100
            print(f"特征减少: {reduction_percent:.1f}%")
        
        print(f"\n特征工程:")
        print(f"使用的方法: {', '.join(self.feature_engineering.feature_info.get('methods_used', []))}")
        print(f"特征数量变化: {original_shape[1]} -> {self.feature_engineering.feature_info.get('engineered_features', 0)}")
        
        print(f"\n特征选择:")
        print(f"最佳方法: {self.feature_selector.best_method}")
        print(f"选择特征数: {selected_shape[1]}")
        print(f"选择得分: {self.feature_selector.best_score:.4f}")
        
        print(f"\n模型优化:")
        print(f"优化的模型数: {len(self.hyperparameter_optimizer.best_models)}")
        
        print(f"\n最佳模型:")
        print(f"名称: {self.best_model_name}")
        print(f"测试准确率: {self.best_score:.4f}")
        
        print(f"\n性能指标:")
        print(f"总执行时间: {self.execution_time:.1f} 秒")
        if self.execution_time > 0:
            print(f"平均每秒处理样本: {original_shape[0]/self.execution_time:.1f}")
        
        print(f"\n保存的文件:")
        print("results/ 目录下:")
        print("  - feature_selection_analysis.png (特征选择分析)")
        print("  - hyperparameter_optimization_results.png (超参数优化结果)")
        print("  - final_model_evaluation.png (最终模型评估)")
        print("  - pipeline_results_*.json (管道详细结果)")
        print("  - model_results_summary_*.csv (模型结果总结)")
        print("  - selected_features_*.csv (选择的特征名称)")
        print("  - new_data_predictions_*.xlsx (新数据预测结果)")
        print("  - new_data_predictions_*.csv (新数据预测结果)")
        print("  - prediction_statistics_*.csv (预测统计信息)")
        
        print("\nmodels/ 目录下:")
        print("  - best_model_*.pkl (最佳模型)")
        print("  - all_models_*.pkl (所有模型)")
        print("  - feature_engineering_*.pkl (特征工程管道)")
        print("  - feature_selector_*.pkl (特征选择器)")
        print("  - hyperparameter_optimizer_*.pkl (超参数优化器)")
        
        print("\n后续步骤建议:")
        if self.best_score < 0.7:
            print("1. 考虑增加训练数据量")
            print("2. 尝试更复杂的特征工程方法")
            print("3. 检查数据质量和标签平衡")
            print("4. 考虑使用深度学习模型")
        elif self.best_score < 0.9:
            print("1. 模型性能良好，可以部署使用")
            print("2. 考虑模型集成进一步提升性能")
            print("3. 定期重新训练模型以适应数据变化")
        else:
            print("1. 模型性能优秀，可以部署到生产环境")
            print("2. 考虑模型压缩和加速")
            print("3. 建立监控和报警机制")
        
        print("\n" + "="*80)
        print("自动化机器学习管道执行完成!")
        print("="*80)

def main():
    """主函数"""
    print("自动化机器学习管道")
    print("="*80)
    print("这个管道将自动执行以下步骤:")
    print("1. 数据加载和预处理")
    print("2. 自动化特征工程")
    print("3. 智能特征选择")
    print("4. 模型超参数优化")
    print("5. 集成模型构建")
    print("6. 模型评估和选择")
    print("7. 结果可视化")
    print("8. 模型保存和部署")
    print("9. 新数据预测")
    print("="*80)
    
    # 创建并运行自动化机器学习管道
    pipeline = AutoMLPipeline(random_state=42)
    pipeline.run()

if __name__ == "__main__":
    main()