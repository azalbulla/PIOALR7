import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')


# загрузка и предобработка
def load_and_preprocess():
    df = sns.load_dataset('titanic')
    print(f'загружено строк: {df.shape[0]}')
    print(f'колонки: {list(df.columns)}')

    df = df.dropna(how='all')

    if 'age' in df.columns and df['age'].isnull().any():
        df['age'] = df['age'].fillna(df['age'].median())
    if 'embarked' in df.columns and df['embarked'].isnull().any():
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    if 'deck' in df.columns and df['deck'].isnull().any():
        df['deck'] = df['deck'].cat.add_categories(['unknown'])
        df['deck'] = df['deck'].fillna('unknown')
    if 'embark_town' in df.columns and df['embark_town'].isnull().any():
        df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype('category')
    if 'class' in df.columns:
        df['class'] = df['class'].astype('category')
    if 'embarked' in df.columns:
        df['embarked'] = df['embarked'].astype('category')

    if 'survived' in df.columns:
        df['survived_label'] = df['survived'].map({0: 'погиб', 1: 'выжил'}).astype('category')

    return df


# seaborn графики
def plot_seaborn(df, output_dir='outputs_titanic'):
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = ['age', 'fare', 'sibsp', 'parch']

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='age', kde=True, bins=30, color='steelblue')
    plt.title('распределение возраста пассажиров')
    plt.xlabel('возраст (лет)')
    plt.ylabel('количество')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seaborn_hist_kde.png'), dpi=200)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='age', y='fare', hue='survived', alpha=0.6)
    plt.title('цена билета от возраста')
    plt.xlabel('возраст (лет)')
    plt.ylabel('цена билета ($)')
    plt.legend(title='статус')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seaborn_scatter.png'), dpi=200)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='class', y='age', hue='sex', split=True, palette='pastel')
    plt.title('распределение возраста по классу и полу')
    plt.xlabel('класс')
    plt.ylabel('возраст (лет)')
    plt.legend(title='пол')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seaborn_violin.png'), dpi=200)
    plt.show()
    plt.close()

    corr = df[numeric_cols + ['survived']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('матрица корреляций')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seaborn_heatmap.png'), dpi=200)
    plt.show()
    plt.close()


# plotly графики
def plot_plotly(df, output_dir='outputs_titanic'):
    os.makedirs(output_dir, exist_ok=True)

    fig_scatter = px.scatter(df, x='age', y='fare', color='survived_label',
                             hover_data=['sex', 'class', 'embark_town'],
                             title='scatterplot: возраст vs цена билета',
                             labels={'age': 'возраст (лет)', 'fare': 'цена ($)'})
    fig_scatter.write_html(os.path.join(output_dir, 'plotly_scatter.html'))
    fig_scatter.show()

    age_avg_fare = df.groupby('age')['fare'].mean().reset_index()
    fig_line = px.line(age_avg_fare, x='age', y='fare', markers=True,
                       title='средняя цена билета по возрастам')
    fig_line.update_layout(xaxis_title='возраст (лет)', yaxis_title='средняя цена ($)')
    fig_line.write_html(os.path.join(output_dir, 'plotly_line.html'))
    fig_line.show()

    survival_by_class = df.groupby(['class', 'survived_label']).size().reset_index(name='count')
    fig_bar = px.bar(survival_by_class, x='class', y='count', color='survived_label',
                     title='выживаемость по классам',
                     labels={'class': 'класс', 'count': 'количество', 'survived_label': 'статус'})
    fig_bar.write_html(os.path.join(output_dir, 'plotly_bar.html'))
    fig_bar.show()

    numeric_cols = ['age', 'fare', 'sibsp', 'parch', 'survived']
    corr = df[numeric_cols].corr()
    fig_heatmap = px.imshow(corr, text_auto='.2f', aspect='auto',
                            color_continuous_scale='rdbu_r',
                            title='интерактивная матрица корреляций')
    fig_heatmap.update_layout(height=500)
    fig_heatmap.write_html(os.path.join(output_dir, 'plotly_heatmap.html'))
    fig_heatmap.show()

    features = ['age', 'fare', 'sibsp', 'parch']
    feature_names = {'age': 'возраст', 'fare': 'цена', 'sibsp': 'sibsp', 'parch': 'parch'}

    fig_dropdown = make_subplots(rows=1, cols=1)
    buttons = []
    for i, feat in enumerate(features):
        fig_dropdown.add_trace(
            go.Histogram(x=df[feat], name=feature_names[feat], visible=(i == 0)),
            row=1, col=1
        )
        visible = [False] * len(features)
        visible[i] = True
        buttons.append(dict(
            method='update',
            args=[{'visible': visible}, {'title': f'распределение: {feature_names[feat]}'}],
            label=feature_names[feat]
        ))

    fig_dropdown.update_layout(
        title='выбор признака (выпадающий список)',
        xaxis_title='значение',
        yaxis_title='частота',
        updatemenus=[dict(buttons=buttons, direction='down', showactive=True,
                          x=0.05, y=1.15, xanchor='left', yanchor='top')],
        height=500
    )
    fig_dropdown.write_html(os.path.join(output_dir, 'plotly_dropdown.html'))
    fig_dropdown.show()


# сравнение библиотек
def print_comparison():
    print("\n" + "=" * 60)
    print("сравнение seaborn и plotly")
    print("=" * 60)
    print("""
    seaborn:
    + простой синтаксис
    + хорош для быстрого анализа
    + статические графики
    - нет интерактивности

    plotly:
    + интерактивность (зум, тултипы)
    + выпадающие списки
    + экспорт в html
    - сложнее синтаксис
    """)
    print("=" * 60)


# запуск
if __name__ == '__main__':
    print("загрузка данных titanic...")
    df = load_and_preprocess()

    print("построение seaborn графиков...")
    plot_seaborn(df)

    print("построение plotly графиков...")
    plot_plotly(df)

    print_comparison()

    print('\nобработка завершена.')
    print('статические графики: outputs_titanic/ .png')
    print('интерактивные графики: outputs_titanic/ .html')
