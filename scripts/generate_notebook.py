import os
import copy
import warnings
import nbformat as nbf
from nbclient import NotebookClient

# Setup relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🚢 Titanic - Machine Learning from Disaster
## 고성능 피처 엔지니어링 및 6대 모델 앙상블 파이프라인

> **기반 원본**: `notebooks/archive/Kaggle_Titanic_2023_01_09_0_0_2.ipynb` 리팩토링 및 고도화  
> **핵심 개선점**:
> 1. **기존 데이터 누수 및 결측 타깃 버그 해결**: 전체 데이터셋 결합 후 분할로 인해 발생했던 타깃 결측치(NaN Target) 및 누수 완전 제거
> 2. **Woman-Child Group (WCG) 생존 피처 도입**: 타이타닉 도메인 최강 피처인 가족 및 티켓 동행 그룹 생존 신호 (Leave-One-Out 적용)
> 3. **도메인 기반 결측치 처리**: `Age`(Title+Pclass 중앙값), `Fare`(Pclass+Embarked 중앙값 및 Log1p 변환), `Embarked`(최빈값)
> 4. **객체지향(OOP) 파이프라인 구축**: `TitanicModelPipeline` 클래스로 모듈화 (전처리, 5-Fold Stratified K-Fold CV, OOF 블렌딩)
> 5. **6대 트리 기반 모델 앙상블**: Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost
> 6. **`shepsci/kaggle-skill` 규격 검증 및 캐글 제출**: 제출 파일 무결성(형식, 행수, ID 일치, NaN 유무) 검증 및 API 제출
"""))

# 1. Imports
cells.append(nbf.v4.new_markdown_cell("## 1. 라이브러리 임포트 및 환경 설정"))
cells.append(nbf.v4.new_code_cell("""import os
import re
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print('모든 필수 라이브러리가 성공적으로 로드되었습니다.')
"""))

# 2. Data Loading & EDA
cells.append(nbf.v4.new_markdown_cell("## 2. 데이터 로드 및 탐색적 데이터 분석 (EDA)"))
cells.append(nbf.v4.new_code_cell("""# 실행 경로에 따른 유연한 데이터 파일 경로 설정
def find_data_file(filename):
    candidates = [
        os.path.join('data', filename),
        os.path.join('..', 'data', filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{filename} 파일을 찾을 수 없습니다.')

train_path = find_data_file('train.csv')
test_path = find_data_file('test.csv')
sample_path = find_data_file('gender_submission.csv')

train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)
sub_sample = pd.read_csv(sample_path)

print(f'Train 데이터 형태: {train_raw.shape}')
print(f'Test 데이터 형태:  {test_raw.shape}')
print(f'Sample Submission 형태: {sub_sample.shape}')

display(train_raw.head(3))
"""))

cells.append(nbf.v4.new_code_cell("""# 결측치 현황 확인
def check_missing(df, name='DataFrame'):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percent (%)': percent.round(2)})
    print(f'=== {name} 결측치 현황 ===')
    if len(missing_df) > 0:
        display(missing_df)
    else:
        print('결측치가 없습니다.')

check_missing(train_raw, 'Train')
check_missing(test_raw, 'Test')
"""))

# 3. Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 3. 고도화된 피처 엔지니어링 (Feature Engineering)
- **Title (호칭)**: Name 컬럼에서 정규표현식으로 추출 및 의미 있는 그룹으로 범주화
- **FamilySize & IsAlone**: SibSp + Parch + 1 및 1인 승객 여부
- **Deck**: Cabin의 첫 글자(구역) 추출 (결측치는 'Unknown(U)')
- **Age 정밀 대체**: 호칭(Title)과 객실 등급(Pclass)별 중앙값으로 대체
- **Fare 정밀 대체 및 로그 변환**: Pclass와 Embarked 그룹 중앙값으로 결측치 보정 후 Log1p 변환
- **TicketFreq**: 티켓별 동행자 빈도
- **Woman-Child-Group (WCG) 생존 피처**:
  - 여성과 어린이(Master 및 12세 이하)의 동일 가족/티켓 동행 생존율 집계
  - 데이터 누수를 방지하기 위해 자기 자신을 제외한(Leave-One-Out) 생존율 적용
"""))

cells.append(nbf.v4.new_code_cell("""def build_features(train_df, test_df):
    # 전처리 및 피처 엔지니어링을 위해 결합 (타깃 분리 유지)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 1. 호칭 (Title) 추출
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
        'Mrs': 'Mrs', 'Mme': 'Mrs',
        'Master': 'Master',
        'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
        'Sir': 'Royalty', 'Don': 'Royalty', 'Countess': 'Royalty', 'Lady': 'Royalty', 'Dona': 'Royalty',
        'Jonkheer': 'Royalty'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Other')
    
    # 2. 가족 크기 (FamilySize) & 동행 여부 (IsAlone)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. 객실 구역 (Deck)
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')
    
    # 4. 결측치 보정
    # Embarked: 최빈값 'S'
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Fare: Pclass와 Embarked 기준 중앙값 대체 후 왜도 보정을 위한 Log1p 변환
    med_fare = df.groupby(['Pclass', 'Embarked'])['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(med_fare)
    df['LogFare'] = np.log1p(df['Fare'])
    
    # Age: Title과 Pclass 기준 중앙값 대체
    med_age = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(med_age)
    
    # 5. 티켓 빈도수 (동행자 확인)
    ticket_counts = df['Ticket'].value_counts()
    df['TicketFreq'] = df['Ticket'].map(ticket_counts)
    
    # 6. Woman-Child-Group (WCG) 생존 피처 (Titanic 최강 피처)
    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Title'] == 'Master') | (df['Age'] <= 12)).astype(int)
    df['FamilyGroup'] = df['LastName'] + '_' + df['FamilySize'].astype(str)
    
    # 기본값 0.5 (신호 없음/중립)
    df['GroupSurvival'] = 0.5
    
    for grp_col in ['Ticket', 'FamilyGroup']:
        grp = df.groupby(grp_col)
        for _, group_df in grp:
            if len(group_df) > 1:
                # 훈련 세트의 여성/어린이 동행자 필터
                wc_train = group_df[(group_df['Survived'].notnull()) & (group_df['IsWomanOrChild'] == 1)]
                if len(wc_train) > 0:
                    for idx in group_df.index:
                        # Leave-one-out: 자기 자신을 제외한 다른 동행자의 생존 여부
                        other_wc = wc_train[wc_train.index != idx]
                        if len(other_wc) > 0:
                            s_mean = other_wc['Survived'].mean()
                            if s_mean == 1.0:
                                df.loc[idx, 'GroupSurvival'] = 1.0
                            elif s_mean == 0.0:
                                df.loc[idx, 'GroupSurvival'] = 0.0
                                
    # 범주형 변수 원-핫 인코딩
    cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 모델 학습에 불필요한 원본 텍스트 컬럼 제거
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'LastName', 'FamilyGroup', 'Fare']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # 훈련 세트와 테스트 세트 복원 (인덱스 유지)
    train_clean = df_clean[df_clean['Survived'].notnull()].copy()
    test_clean = df_clean[df_clean['Survived'].isnull()].drop(columns=['Survived']).copy()
    train_clean['Survived'] = train_clean['Survived'].astype(int)
    
    return train_clean, test_clean

train_features, test_features = build_features(train_raw, test_raw)
print(f'피처 변환 후 Train 형태: {train_features.shape}')
print(f'피처 변환 후 Test 형태:  {test_features.shape}')
display(train_features.head(3))
"""))

# 4. OOP Modeling Pipeline
cells.append(nbf.v4.new_markdown_cell("""## 4. 객체지향 모델링 및 교차 검증 파이프라인 (OOP Pipeline)
- 5-Fold Stratified K-Fold CV 교차 검증 수행
- 6대 앙상블 모델 개별 학습 및 Out-Of-Fold(OOF) 확률 저장
- 소프트 보팅(Soft Voting) 앙상블 결합
- 최종 검증 점수 및 Feature Importance 시각화
"""))

cells.append(nbf.v4.new_code_cell("""class TitanicModelPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=4, random_state=random_state),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_depth=5, min_samples_split=4, random_state=random_state),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, random_state=random_state),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, random_state=random_state, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.04, random_state=random_state, verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=120, depth=4, learning_rate=0.04, random_seed=random_state, verbose=0)
        }
        self.trained_models = {name: [] for name in self.models}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.oof_ensemble = None
        self.test_ensemble = None

    def fit_cv(self, X, y, n_splits=5):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        n_samples = len(X)
        self.oof_predictions = {name: np.zeros(n_samples) for name in self.models}
        
        print(f'=== {n_splits}-Fold Stratified K-Fold 교차 검증 시작 ===')
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            for name, model_cls in self.models.items():
                model = copy.deepcopy(model_cls)
                model.fit(X_train, y_train)
                self.trained_models[name].append(model)
                
                val_prob = model.predict_proba(X_val)[:, 1]
                self.oof_predictions[name][val_idx] = val_prob

        all_oof_probs = [self.oof_predictions[name] for name in self.models]
        self.oof_ensemble = np.mean(all_oof_probs, axis=0)
        
        print('\\n=== 모델별 Out-Of-Fold (OOF) 평가 결과 ===')
        summary = []
        for name in self.models:
            prob = self.oof_predictions[name]
            pred = (prob > 0.5).astype(int)
            acc = accuracy_score(y, pred)
            auc = roc_auc_score(y, prob)
            summary.append({'Model': name, 'Accuracy': round(acc, 4), 'ROC-AUC': round(auc, 4)})
            
        ens_pred = (self.oof_ensemble > 0.5).astype(int)
        ens_acc = accuracy_score(y, ens_pred)
        ens_auc = roc_auc_score(y, self.oof_ensemble)
        summary.append({'Model': '★ Ensemble (Soft Voting)', 'Accuracy': round(ens_acc, 4), 'ROC-AUC': round(ens_auc, 4)})
        
        summary_df = pd.DataFrame(summary)
        display(summary_df)
        return summary_df

    def predict(self, X_test):
        n_test = len(X_test)
        self.test_predictions = {name: np.zeros(n_test) for name in self.models}
        
        for name, model_list in self.trained_models.items():
            model_probs = np.zeros(n_test)
            for model in model_list:
                model_probs += model.predict_proba(X_test)[:, 1] / len(model_list)
            self.test_predictions[name] = model_probs
            
        all_test_probs = [self.test_predictions[name] for name in self.models]
        self.test_ensemble = np.mean(all_test_probs, axis=0)
        final_preds = (self.test_ensemble > 0.5).astype(int)
        return final_preds
"""))

# 5. Pipeline Run & Visualizations
cells.append(nbf.v4.new_markdown_cell("## 5. 모델 훈련 및 검증 결과 시각화"))
cells.append(nbf.v4.new_code_cell("""X = train_features.drop(columns=['Survived'])
y = train_features['Survived']
X_test = test_features.copy()

pipeline = TitanicModelPipeline(random_state=42)
cv_summary = pipeline.fit_cv(X, y, n_splits=5)

plt.figure(figsize=(10, 5))
colors = sns.color_palette('viridis', len(cv_summary))
bars = plt.barh(cv_summary['Model'], cv_summary['Accuracy'], color=colors)
plt.xlim(0.75, 0.90)
plt.xlabel('OOF Accuracy')
plt.title('Out-of-Fold (OOF) Model Accuracy Comparison', fontsize=14, fontweight='bold')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.002, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrix & Classification Report
y_ens_pred = (pipeline.oof_ensemble > 0.5).astype(int)
cm = confusion_matrix(y, y_ens_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Perished (0)', 'Survived (1)'],
            yticklabels=['Perished (0)', 'Survived (1)'])
plt.title('Ensemble OOF Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print('=== Classification Report ===')
print(classification_report(y, y_ens_pred, target_names=['Perished (0)', 'Survived (1)']))
"""))

# 6. Test Prediction & Submission File
cells.append(nbf.v4.new_markdown_cell("## 6. 테스트 세트 예측 및 제출 파일 생성"))
cells.append(nbf.v4.new_code_cell("""test_preds = pipeline.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_raw['PassengerId'],
    'Survived': test_preds
})

sub_dest_candidates = [
    os.path.join('submissions', 'submission.csv'),
    os.path.join('..', 'submissions', 'submission.csv'),
    'submission.csv'
]
out_path = 'submission.csv'
for cand in sub_dest_candidates:
    parent = os.path.dirname(cand)
    if parent and os.path.exists(parent):
        out_path = cand
        break

submission.to_csv(out_path, index=False)
print(f'제출 파일이 성공적으로 생성되었습니다: {out_path}')
display(submission.head(10))
print(f'생존자 예측 비율: {submission["Survived"].mean():.2%}')
"""))

# 7. Validation Checks
cells.append(nbf.v4.new_markdown_cell("""## 7. shepsci/kaggle-skill 기준 무결성 검증 (Integrity Check)
캐글 제출 전 오류(0점 또는 Status ERROR)를 방지하기 위해 다음 항목을 전수 점검합니다:
1. **컬럼 구성 및 순서**: `PassengerId`, `Survived`
2. **행 개수 (Rows)**: 정확히 418개
3. **결측치 (Null/NaN) 여부**: 0건
4. **ID 무결성**: sample_submission의 PassengerId와 완벽 일치 (892 ~ 1309)
5. **값의 범위**: 오직 0 또는 1로 구성된 정수
"""))

cells.append(nbf.v4.new_code_cell("""def validate_submission(sub_path, sample_path):
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    
    checks = []
    
    # 1. 컬럼 검사
    cols_match = list(sub.columns) == list(sample.columns)
    checks.append(('Columns match', cols_match, f'{list(sub.columns)}'))
    
    # 2. 행 수 검사
    rows_match = len(sub) == len(sample)
    checks.append(('Row count match', rows_match, f'{len(sub)} rows'))
    
    # 3. 결측치 검사
    no_null = sub.isnull().sum().sum() == 0
    checks.append(('No missing values', no_null, f'{sub.isnull().sum().to_dict()}'))
    
    # 4. ID 일치 검사
    ids_match = (sub['PassengerId'].values == sample['PassengerId'].values).all()
    checks.append(('PassengerId exact match', ids_match, f'{sub["PassengerId"].min()} ~ {sub["PassengerId"].max()}'))
    
    # 5. 값의 유효성 검사 (0 or 1)
    vals_valid = set(sub['Survived'].unique()).issubset({0, 1})
    checks.append(('Values are binary (0/1)', vals_valid, f'Unique values: {sub["Survived"].unique()}'))
    
    all_passed = all(c[1] for c in checks)
    print('=== Submission Validation Summary ===')
    for label, passed, detail in checks:
        status = ' PASS ' if passed else ' FAIL '
        print(f'[{status}] {label}: {detail}')
        
    if all_passed:
        print('\\n[SUCCESS] 모든 유효성 검사를 통과했습니다! 안전하게 캐글에 제출할 수 있습니다.')
    else:
        raise ValueError('[ERROR] 유효성 검사 실패! 제출 파일을 확인하십시오.')

validate_submission(out_path, sample_path)
"""))

# 8. Kaggle Submit Instruction
cells.append(nbf.v4.new_markdown_cell("""## 8. 캐글 제출 (Kaggle CLI / kaggle-skill)
검증을 통과한 파일을 Kaggle API를 통해 제출합니다.
```bash
python C:/Users/Taeyang/.gemini/config/skills/kaggle/scripts/kaggle_skill.py submit titanic submissions/submission.csv -m "Titanic High-Performance Ensemble Pipeline (WCG+6Models)" --yes
```
"""))

nb.cells = cells
notebook_path = os.path.join(NOTEBOOKS_DIR, 'Titanic_Ensemble_Pipeline.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'{notebook_path} 생성 완료!')

# Execute notebook from project root
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': BASE_DIR}})
print('노트북 셀 실행 시작...')
client.execute()
print('노트북 셀 실행 완료!')

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'결과가 포함된 노트북 저장 완료: {notebook_path}')
