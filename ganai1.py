import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义简单的参数类
class Args:
    def __init__(self, alpha=0.1, k=10):
        self.alpha = alpha
        self.k = k

def load_data():
    """加载数据并转换为numpy数组"""
    try:
        train_features = pd.read_excel('z1/Xtrain.xlsx')
        train_labels = pd.read_excel('z1/ytrain.xlsx').values.ravel()
        test_features = pd.read_excel('z1/Xtest.xlsx')
        test_labels = pd.read_excel('z1/ytest.xlsx').values.ravel()
        
        print("数据加载成功:")
        print(f"训练集: {train_features.shape[0]} 个样本, {train_features.shape[1]} 个特征")
        print(f"测试集: {test_features.shape[0]} 个样本, {test_features.shape[1]} 个特征")
        
        # 转换为numpy数组
        X_train = train_features.values
        X_test = test_features.values
        
        return X_train, train_labels, X_test, test_labels, train_features.columns.tolist()
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None, None, None

def load_new_data():
    """加载新数据集进行预测"""
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

class Selection:
    """特征选择类，基于Lasso回归进行特征筛选"""
    def __init__(self, method='lasso', alpha=0.1, k=10):
        self.method = method
        self.alpha = alpha
        self.k = k
        self.mb_ = None  # 选中的特征索引
        self.scaler = StandardScaler()  # 添加标准化器

    def fit(self, X, y=None):
        """使用Lasso回归进行特征选择（最终使用）"""
        print("\n===== 最终特征选择 =====")
        
        # 标准化数据（Lasso对特征缩放敏感）
        X_scaled = self.scaler.fit_transform(X)
        
        cls = Lasso(alpha=self.alpha, max_iter=10000)
        cls.fit(X_scaled, y)
        
        # 计算特征重要性阈值
        if self.k > 0 and self.k <= X.shape[1]:
            # 选择绝对值最大的k个系数对应的特征
            coef_abs = np.abs(cls.coef_)
            threshold = np.sort(coef_abs)[-self.k]
            
            # 选择重要性高于阈值的特征
            sel = SelectFromModel(cls, prefit=True, threshold=threshold, max_features=self.k)
            self.mb_ = np.array(list(range(X.shape[1])))[sel.get_support()]
            
            print(f"特征重要性阈值: {threshold:.4f}")
        else:
            # 如果没有指定k，选择所有非零系数特征
            self.mb_ = np.where(cls.coef_ != 0)[0]
            print("使用非零系数特征选择")
        
        print(f"从 {X.shape[1]} 个特征中选择了 {len(self.mb_)} 个特征")
        print("选择的特征索引:", self.mb_)
        
        return self

    def transform(self, X):
        """应用特征选择结果"""
        if self.mb_ is None or len(self.mb_) == 0:
            return X
        # 先标准化，然后选择特征
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.mb_]
    
    def get_support(self, indices=False):
        """获取选择的特征索引"""
        if indices:
            return self.mb_
        else:
            support = np.zeros(self.scaler.mean_.shape[0], dtype=bool)
            support[self.mb_] = True
            return support

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    """训练和评估多种模型"""
    print("\n" + "="*50)
    print("STEP 1: 特征选择")
    print("="*50)
    
    # 创建特征选择器
    selector = Selection(method='lasso', alpha=0.1, k=10)
    selector.fit(X_train, y_train)
    
    # 应用特征选择
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"特征选择后训练集维度: {X_train_selected.shape}")
    print(f"特征选择后测试集维度: {X_test_selected.shape}")
    
    # 获取选择的特征名称
    selected_feature_names = [feature_names[i] for i in selector.mb_] if len(selector.mb_) > 0 else []
    
    print("\n" + "="*50)
    print("STEP 2: 训练和评估多种模型")
    print("="*50)
    
    # 定义多种分类算法
    models = {
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
        '支持向量机': SVC(kernel='rbf', probability=True, random_state=42),
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K近邻': KNeighborsClassifier(n_neighbors=5),
        '朴素贝叶斯': GaussianNB(),
        '决策树': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n--- 训练 {name} ---")
        
        # 训练模型
        model.fit(X_train_selected, y_train)
        
        # 训练集预测
        y_train_pred = model.predict(X_train_selected)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # 测试集预测
        y_test_pred = model.predict(X_test_selected)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # 交叉验证得分
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"交叉验证准确率: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # 保存结果
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }
        
        # 更新最佳模型
        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_model_name = name
    
    print("\n" + "="*50)
    print("模型性能比较")
    print("="*50)
    
    # 创建性能比较表格
    comparison_df = pd.DataFrame({
        '模型': list(results.keys()),
        '训练集准确率': [results[name]['train_acc'] for name in results.keys()],
        '测试集准确率': [results[name]['test_acc'] for name in results.keys()],
        '交叉验证平均': [results[name]['cv_mean'] for name in results.keys()],
        '交叉验证标准差': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('测试集准确率', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    print(f"\n最佳模型: {best_model_name} (测试集准确率: {best_score:.4f})")
    
    # 可视化模型性能比较
    visualize_model_comparison(results)
    
    # 可视化特征重要性（对于有特征重要性的模型）
    visualize_feature_importance(models, selector, selected_feature_names, X_train_selected, y_train)
    
    return best_model, selector, results, best_model_name

def visualize_model_comparison(results):
    """可视化模型性能比较"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 模型准确率比较柱状图
    models_names = list(results.keys())
    train_accs = [results[name]['train_acc'] for name in models_names]
    test_accs = [results[name]['test_acc'] for name in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_accs, width, label='训练集', color='skyblue')
    axes[0, 0].bar(x + width/2, test_accs, width, label='测试集', color='lightcoral')
    axes[0, 0].set_xlabel('模型')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_title('不同模型的训练集和测试集准确率比较')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 交叉验证性能比较
    cv_means = [results[name]['cv_mean'] for name in models_names]
    cv_stds = [results[name]['cv_std'] for name in models_names]
    
    axes[0, 1].bar(models_names, cv_means, yerr=cv_stds, capsize=5, color='lightgreen')
    axes[0, 1].set_xlabel('模型')
    axes[0, 1].set_ylabel('交叉验证准确率')
    axes[0, 1].set_title('模型交叉验证性能比较')
    axes[0, 1].set_xticklabels(models_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最佳模型的混淆矩阵
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_acc'])
    y_test_pred = results[best_model_name]['y_test_pred']
    
    # 为了画混淆矩阵，需要知道真实的y_test
    # 这里假设y_test是已知的（在train_and_evaluate_models函数外部）
    # 所以这个图在main函数中画
    
    axes[0, 1].text(0.02, 0.98, f'最佳模型: {best_model_name}', 
                   transform=axes[0, 1].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 模型性能热图
    performance_data = np.array([train_accs, test_accs, cv_means]).T
    sns.heatmap(performance_data, annot=True, fmt='.3f', 
                xticklabels=['训练集', '测试集', '交叉验证'],
                yticklabels=models_names, ax=axes[1, 0], cmap='YlOrRd')
    axes[1, 0].set_title('模型性能热图')
    
    # 5. 模型性能折线图
    for i, name in enumerate(models_names):
        axes[1, 1].plot([0, 1, 2], [train_accs[i], test_accs[i], cv_means[i]], 
                       marker='o', label=name, linewidth=2)
    axes[1, 1].set_xlabel('评估指标')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_title('模型在不同评估指标上的表现')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['训练集', '测试集', '交叉验证'])
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n模型比较图已保存为 'model_comparison.png'")
    plt.show()

def visualize_feature_importance(models, selector, feature_names, X_train, y_train):
    """可视化特征重要性"""
    # 只可视化有特征重要性的模型
    models_with_importance = ['随机森林', '梯度提升', '决策树']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model_name in enumerate(models_with_importance):
        if model_name in models:
            # 重新训练模型以获取特征重要性
            model = models[model_name]
            model.fit(X_train, y_train)
            
            # 检查模型是否有feature_importances_属性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # 只显示前10个最重要的特征
                top_n = min(10, len(importances))
                top_indices = indices[:top_n]
                
                axes[idx].bar(range(top_n), importances[top_indices])
                axes[idx].set_xticks(range(top_n))
                
                # 使用特征名称或索引
                if feature_names and len(feature_names) >= len(importances):
                    top_names = [feature_names[i] for i in top_indices]
                    axes[idx].set_xticklabels(top_names, rotation=45, ha='right', fontsize=8)
                else:
                    axes[idx].set_xticklabels([f'特征{i}' for i in top_indices], rotation=45, ha='right')
                
                axes[idx].set_xlabel('特征')
                axes[idx].set_ylabel('重要性')
                axes[idx].set_title(f'{model_name}特征重要性')
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("特征重要性比较图已保存为 'feature_importance_comparison.png'")
    plt.show()
    
def predict_new_data(best_model, selector, feature_names):
    """使用最佳模型预测新数据"""
    print("\n" + "="*50)
    print("新数据预测")
    print("="*50)

    # 加载新数据
    X_new, new_feature_names = load_new_data()
    
    if X_new is not None:
        # 检查特征数量是否匹配
        if X_new.shape[1] != len(feature_names):
            print(f"警告: 新数据特征数({X_new.shape[1]})与训练数据特征数({len(feature_names)})不匹配!")
            # 尝试调整特征数量
            if X_new.shape[1] > len(feature_names):
                X_new = X_new[:, :len(feature_names)]
                print(f"已截取前{len(feature_names)}个特征")
            else:
                # 填充零值
                padding = np.zeros((X_new.shape[0], len(feature_names) - X_new.shape[1]))
                X_new = np.hstack([X_new, padding])
                print(f"已填充零值到{len(feature_names)}个特征")
        
        # 应用特征选择
        X_new_selected = selector.transform(X_new)
        
        # 预测
        predictions = best_model.predict(X_new_selected)
        
        # 如果有predict_proba方法，也获取概率
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_new_selected)
        
        # 保存预测结果
        results_df = pd.DataFrame({
            '样本ID': range(1, len(predictions) + 1),
            '预测类别': predictions
        })
        
        # 添加概率列
        if hasattr(best_model, 'predict_proba'):
            for i in range(probabilities.shape[1]):
                results_df[f'类别{i}概率'] = probabilities[:, i]
        
        # 保存到Excel
        results_df.to_excel('new_data_predictions.xlsx', index=False)
        print(f"新数据预测结果已保存到 'new_data_predictions.xlsx'")
        
        # 显示预测结果统计
        print(f"\n预测结果统计:")
        print(f"总样本数: {len(predictions)}")
        
        unique_classes, class_counts = np.unique(predictions, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            percentage = count / len(predictions) * 100
            print(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)")
        
        return predictions, results_df
    else:
        print("无新数据可供预测")
        return None, None

def save_models(best_model, selector, best_model_name, results):
    """保存模型和特征选择器"""
    try:
        # 保存最佳模型
        best_model_filename = f'best_model_{best_model_name}.pkl'
        joblib.dump(best_model, best_model_filename)
        print(f"最佳模型已保存到: {best_model_filename}")
        
        # 保存特征选择器
        selector_filename = 'feature_selector.pkl'
        joblib.dump(selector, selector_filename)
        print(f"特征选择器已保存到: {selector_filename}")
        
        # 保存所有模型结果
        results_filename = 'all_models_results.pkl'
        joblib.dump(results, results_filename)
        print(f"所有模型结果已保存到: {results_filename}")
        
        # 保存性能比较表格
        comparison_data = {
            'model_names': list(results.keys()),
            'train_accs': [results[name]['train_acc'] for name in results.keys()],
            'test_accs': [results[name]['test_acc'] for name in results.keys()],
            'cv_means': [results[name]['cv_mean'] for name in results.keys()],
            'cv_stds': [results[name]['cv_std'] for name in results.keys()]
        }
        joblib.dump(comparison_data, 'model_comparison_data.pkl')
        
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def plot_confusion_matrix_for_best_model(best_model_name, results, y_test):
    """绘制最佳模型的混淆矩阵"""
    if best_model_name in results:
        y_pred = results[best_model_name]['y_test_pred']
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(np.unique(y_test)),
                   yticklabels=sorted(np.unique(y_test)))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{best_model_name}混淆矩阵')
        plt.tight_layout()
        plt.savefig('best_model_confusion_matrix.png', dpi=300)
        print("最佳模型混淆矩阵已保存为 'best_model_confusion_matrix.png'")
        plt.show()

def main():
    """主函数"""
    print("开始机器学习流程...")
    
    # 1. 加载数据
    X_train, y_train, X_test, y_test, feature_names = load_data()
    
    if X_train is None:
        print("数据加载失败，请检查文件路径和格式！")
        return
    
    # 2. 训练和评估多种模型
    best_model, selector, results, best_model_name = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    # 3. 绘制最佳模型的混淆矩阵
    plot_confusion_matrix_for_best_model(best_model_name, results, y_test)
    
    # 4. 使用最佳模型预测新数据
    predictions, results_df = predict_new_data(best_model, selector, feature_names)
    
    # 5. 保存模型
    save_models(best_model, selector, best_model_name, results)
    
    print("\n" + "="*50)
    print("流程完成！")
    print("="*50)

# 执行主函数
if __name__ == "__main__":
    main()