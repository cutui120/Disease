import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split,
    learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ComprehensiveEvaluator:
    """综合评估器，提供全面的模型评估指标"""
    
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
        """全面评估模型性能"""
        print(f"\n{'='*60}")
        print(f"模型评估: {model_name}")
        print('='*60)
        
        # 1. 基础预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 2. 计算基础指标
        metrics = {
            '训练集': {
                '准确率': accuracy_score(y_train, y_train_pred),
                '精确率': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                '召回率': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                'F1分数': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            },
            '测试集': {
                '准确率': accuracy_score(y_test, y_test_pred),
                '精确率': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                '召回率': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                'F1分数': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }
        }
        
        # 3. ROC曲线和AUC（针对二分类问题）
        try:
            if len(np.unique(y_train)) == 2 and hasattr(model, 'predict_proba'):
                y_test_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_prob)
                metrics['测试集']['AUC'] = roc_auc
                print(f"AUC: {roc_auc:.4f}")
        except:
            print("注意: 无法计算AUC（可能为多分类或模型不支持概率预测）")
        
        # 4. 交叉验证评估
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='accuracy')
        metrics['交叉验证'] = {
            '平均准确率': cv_scores.mean(),
            '标准差': cv_scores.std(),
            '最小': cv_scores.min(),
            '最大': cv_scores.max()
        }
        
        # 5. 打印详细报告
        print("\n详细分类报告:")
        print("测试集:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        
        print("\n性能指标汇总:")
        for dataset, dataset_metrics in metrics.items():
            print(f"\n{dataset}:")
            for metric_name, value in dataset_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
        
        # 6. 返回评估结果
        evaluation_results = {
            'model': model,
            'y_test_pred': y_test_pred,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        return evaluation_results

class FeatureSelector:
    """增强的特征选择器，提供多种选择方法和可解释性"""
    
    def __init__(self, method='random_forest', n_features=10, random_state=42):
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices_ = None
        self.feature_importances_ = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """训练特征选择器"""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'random_forest':
            selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector.fit(X_scaled, y)
            importances = selector.feature_importances_
            
        elif self.method == 'lasso':
            from sklearn.linear_model import LassoCV
            selector = LassoCV(cv=5, random_state=self.random_state)
            selector.fit(X_scaled, y)
            importances = np.abs(selector.coef_)
            
        elif self.method == 'gradient_boosting':
            selector = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
            selector.fit(X_scaled, y)
            importances = selector.feature_importances_
            
        else:
            raise ValueError(f"不支持的特征选择方法: {self.method}")
        
        # 选择最重要的特征
        self.feature_importances_ = importances
        top_indices = np.argsort(importances)[::-1][:self.n_features]
        self.selected_indices_ = top_indices
        
        print(f"特征选择完成: 从 {X.shape[1]} 个特征中选择 {len(top_indices)} 个最重要的特征")
        return self
    
    def transform(self, X):
        """转换数据"""
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_indices_]
    
    def get_selected_features(self, feature_names=None):
        """获取选择的特征名称"""
        if feature_names is None:
            return self.selected_indices_
        return [feature_names[i] for i in self.selected_indices_]
    
    def plot_feature_importance(self, feature_names=None, top_n=15):
        """可视化特征重要性"""
        if self.feature_importances_ is None:
            print("尚未训练特征选择器")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 获取重要性排序
        sorted_idx = np.argsort(self.feature_importances_)[::-1]
        sorted_importances = self.feature_importances_[sorted_idx]
        
        # 限制显示数量
        if len(sorted_idx) > top_n:
            sorted_idx = sorted_idx[:top_n]
            sorted_importances = sorted_importances[:top_n]
        
        # 创建标签
        if feature_names is not None:
            labels = [feature_names[i] for i in sorted_idx]
        else:
            labels = [f'Feature {i}' for i in sorted_idx]
        
        # 绘制条形图
        plt.bar(range(len(sorted_idx)), sorted_importances)
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title(f'特征重要性 ({self.method})')
        plt.xticks(range(len(sorted_idx)), labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class ModelPipeline:
    """完整的模型管道，包含训练、评估、解释和预测"""
    
    def __init__(self, feature_names=None, random_state=42):
        self.feature_names = feature_names
        self.random_state = random_state
        self.best_model = None
        self.selector = None
        self.models_evaluation = {}
        self.evaluator = ComprehensiveEvaluator()
        
    def load_data(self):
        """加载数据"""
        try:
            train_features = pd.read_excel('z1/Xtrain.xlsx')
            train_labels = pd.read_excel('z1/ytrain.xlsx').values.ravel()
            test_features = pd.read_excel('z1/Xtest.xlsx')
            test_labels = pd.read_excel('z1/ytest.xlsx').values.ravel()
            
            print("数据加载成功:")
            print(f"训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 个特征")
            print(f"测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 个特征")
            
            return (train_features.values, train_labels, 
                   test_features.values, test_labels, 
                   train_features.columns.tolist())
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None, None, None, None, None
    
    def load_new_data(self):
        """加载新数据"""
        try:
            new_features = pd.read_excel('z1/Xnew.xlsx')
            print(f"新数据集加载成功: {new_features.shape[0]} 个样本, {new_features.shape[1]} 个特征")
            return new_features.values, new_features.columns.tolist()
        except Exception as e:
            print(f"新数据集加载失败: {e}")
            return None, None
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """训练多个模型并进行综合评估"""
        print("\n" + "="*60)
        print("开始训练和评估多个模型")
        print("="*60)
        
        # 1. 特征选择
        print("\n>>> 步骤1: 特征选择")
        self.selector = FeatureSelector(method='random_forest', n_features=15)
        self.selector.fit(X_train, y_train)
        
        # 应用特征选择
        X_train_selected = self.selector.transform(X_train)
        X_test_selected = self.selector.transform(X_test)
        
        # 2. 定义要训练的模型
        models = {
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            '逻辑回归': LogisticRegression(max_iter=1000, random_state=self.random_state),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            '支持向量机': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'K近邻': KNeighborsClassifier(n_neighbors=5),
            '朴素贝叶斯': GaussianNB(),
            '决策树': DecisionTreeClassifier(random_state=self.random_state, max_depth=5)
        }
        
        # 3. 训练和评估每个模型
        self.models_evaluation = {}
        for name, model in models.items():
            print(f"\n>>> 训练模型: {name}")
            
            # 训练模型
            model.fit(X_train_selected, y_train)
            
            # 综合评估
            eval_results = self.evaluator.evaluate_model(
                model, X_train_selected, y_train, X_test_selected, y_test, name
            )
            
            self.models_evaluation[name] = eval_results
        
        # 4. 选择最佳模型
        self.select_best_model()
        
        # 5. 可视化所有模型性能
        self.visualize_all_models_performance(X_test_selected, y_test)
        
        # 6. 分析泛化能力
        self.analyze_generalization(X_train_selected, y_train)
        
        return self.best_model
    
    def select_best_model(self):
        """选择最佳模型"""
        best_score = 0
        self.best_model_name = None
        
        for name, results in self.models_evaluation.items():
            test_acc = results['metrics']['测试集']['准确率']
            if test_acc > best_score:
                best_score = test_acc
                self.best_model = results['model']
                self.best_model_name = name
        
        print(f"\n{'='*60}")
        print(f"最佳模型: {self.best_model_name}")
        print(f"测试集准确率: {best_score:.4f}")
        print("="*60)
    
    def visualize_all_models_performance(self, X_test, y_test):
        """可视化所有模型的性能对比"""
        model_names = list(self.models_evaluation.keys())
        
        # 准备数据
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        test_scores = np.zeros((len(model_names), len(metrics)))
        
        for i, name in enumerate(model_names):
            results = self.models_evaluation[name]
            test_scores[i, 0] = results['metrics']['测试集']['准确率']
            test_scores[i, 1] = results['metrics']['测试集']['精确率']
            test_scores[i, 2] = results['metrics']['测试集']['召回率']
            test_scores[i, 3] = results['metrics']['测试集']['F1分数']
        
        # 绘制性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 准确率对比
        x = np.arange(len(model_names))
        axes[0, 0].bar(x, test_scores[:, 0], color='skyblue')
        axes[0, 0].set_xlabel('模型')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_title('模型准确率对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 精确率/召回率对比
        width = 0.35
        axes[0, 1].bar(x - width/2, test_scores[:, 1], width, label='精确率', color='lightcoral')
        axes[0, 1].bar(x + width/2, test_scores[:, 2], width, label='召回率', color='lightgreen')
        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('分数')
        axes[0, 1].set_title('模型精确率 vs 召回率')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. F1分数对比
        axes[1, 0].bar(x, test_scores[:, 3], color='gold')
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('F1分数')
        axes[1, 0].set_title('模型F1分数对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 交叉验证性能
        cv_means = []
        cv_stds = []
        for name in model_names:
            cv_scores = self.models_evaluation[name]['cv_scores']
            cv_means.append(cv_scores.mean())
            cv_stds.append(cv_scores.std())
        
        axes[1, 1].bar(x, cv_means, yerr=cv_stds, capsize=5, color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('模型')
        axes[1, 1].set_ylabel('交叉验证准确率')
        axes[1, 1].set_title('模型交叉验证性能（均值和标准差）')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. 混淆矩阵（最佳模型）
        self.plot_confusion_matrix()
        
        # 6. ROC曲线（如果是二分类）
        if len(np.unique(y_test)) == 2:
            self.plot_roc_curves(X_test, y_test)
    
    def plot_confusion_matrix(self):
        """绘制最佳模型的混淆矩阵"""
        if self.best_model_name in self.models_evaluation:
            cm = self.models_evaluation[self.best_model_name]['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'{self.best_model_name} - 混淆矩阵')
            plt.tight_layout()
            plt.savefig('best_model_confusion_matrix.png', dpi=300)
            plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """绘制所有支持概率预测的模型的ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.models_evaluation.items():
            model = results['model']
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('ROC曲线对比')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves_comparison.png', dpi=300)
        plt.show()
    
    def analyze_generalization(self, X_train, y_train):
        """分析模型的泛化能力"""
        print("\n>>> 分析模型泛化能力")
        
        # 学习曲线
        if self.best_model is not None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, test_scores = learning_curve(
                self.best_model, X_train, y_train, train_sizes=train_sizes,
                cv=5, scoring='accuracy', n_jobs=-1
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='训练集')
            plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', label='验证集')
            plt.xlabel('训练样本数量')
            plt.ylabel('准确率')
            plt.title(f'{self.best_model_name} - 学习曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('learning_curve.png', dpi=300)
            plt.show()
    
    def analyze_interpretability(self, X_train, y_train):
        """分析模型的可解释性"""
        print("\n>>> 分析模型可解释性")
        
        if self.selector is not None and self.best_model is not None:
            # 特征重要性
            self.selector.plot_feature_importance(self.feature_names)
            
            # 对于树模型，显示特征重要性
            if hasattr(self.best_model, 'feature_importances_'):
                X_train_selected = self.selector.transform(X_train)
                selected_features = self.selector.get_selected_features(self.feature_names)
                
                plt.figure(figsize=(12, 6))
                importances = self.best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(selected_features)), importances[indices])
                plt.xlabel('特征')
                plt.ylabel('重要性')
                plt.title(f'{self.best_model_name} - 特征重要性')
                plt.xticks(range(len(selected_features)), 
                          [selected_features[i] for i in indices], 
                          rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
    
    def predict_new_data(self):
        """预测新数据"""
        print("\n" + "="*60)
        print("新数据预测")
        print("="*60)
        
        # 加载新数据
        X_new, new_feature_names = self.load_new_data()
        
        if X_new is None:
            print("无新数据可供预测")
            return None
        
        if self.selector is None or self.best_model is None:
            print("请先训练模型")
            return None
        
        # 应用特征选择
        X_new_selected = self.selector.transform(X_new)
        
        # 直接预测类别（不输出概率）
        predictions = self.best_model.predict(X_new_selected)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            '样本ID': range(1, len(predictions) + 1),
            '预测类别': predictions
        })
        
        # 保存结果
        results_df.to_excel('new_data_predictions.xlsx', index=False)
        print(f"预测结果已保存到 'new_data_predictions.xlsx'")
        
        # 打印预测结果
        print("\n预测结果:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def save_models(self):
        """保存模型和选择器"""
        if self.best_model is not None and self.selector is not None:
            # 保存最佳模型
            joblib.dump(self.best_model, f'best_model_{self.best_model_name}.pkl')
            
            # 保存特征选择器
            joblib.dump(self.selector, 'feature_selector.pkl')
            
            # 保存所有评估结果
            joblib.dump(self.models_evaluation, 'all_models_evaluation.pkl')
            
            print(f"\n模型已保存:")
            print(f"- 最佳模型: best_model_{self.best_model_name}.pkl")
            print(f"- 特征选择器: feature_selector.pkl")
            print(f"- 评估结果: all_models_evaluation.pkl")
    
    def run(self):
        """运行完整管道"""
        print("开始运行机器学习管道...")
        
        # 1. 加载数据
        X_train, y_train, X_test, y_test, feature_names = self.load_data()
        if X_train is None:
            return
        
        self.feature_names = feature_names
        
        # 2. 训练和评估模型
        self.train_models(X_train, y_train, X_test, y_test)
        
        # 3. 分析可解释性
        self.analyze_interpretability(X_train, y_train)
        
        # 4. 预测新数据
        self.predict_new_data()
        
        # 5. 保存模型
        self.save_models()
        
        print("\n" + "="*60)
        print("管道运行完成!")
        print("="*60)

# 主程序
if __name__ == "__main__":
    # 创建并运行管道
    pipeline = ModelPipeline(random_state=42)
    pipeline.run()