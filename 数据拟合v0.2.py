import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib import rcParams
import re  # 引入正则表达式模块


# 定义鼠标点击事件处理函数
def on_click(event: MouseEvent):
    if event.inaxes:  # 确保点击在图像区域内
        x_click = event.xdata  # 获取点击的 x 坐标
        y_click = None

        try:
            # 使用最佳拟合模型的结果
            x_fit, y_fit, label = best_fit
            if min(x_fit) <= x_click <= max(x_fit):  # 确保 x_click 在拟合曲线范围内
                y_click = np.interp(x_click, x_fit, y_fit)  # 插值计算 y 值
        except Exception as e:
            print(f"鼠标点击查询失败: {e}")
            return

        if y_click is not None:
            print(f"点击坐标: x = {x_click:.4f}, y = {y_click:.4f}")
            # 在图像上标记点击点
            plt.scatter([x_click], [y_click], color='green', label=f'查询点: ({x_click:.4f}, {y_click:.4f})')
            plt.legend()
            plt.draw()  # 更新图像
        else:
            print("点击的 x 坐标不在最佳拟合曲线范围内。")

# 输入 x，输出拟合曲线的 y 值
def query_y_from_x(x_input, fit_choice, x_data, y_data):
    try:
        x_fit, y_fit, label = perform_fitting(fit_choice, x_data, y_data)
        if min(x_fit) <= x_input <= max(x_fit):
            y_output = np.interp(x_input, x_fit, y_fit)  # 插值计算 y 值
            print(f"拟合模型: {label}")
            print(f"输入 x = {x_input:.4f}, 输出 y = {y_output:.4f}")
        else:
            print("输入的 x 超出了拟合曲线的范围。")
    except Exception as e:
        print(f"查询失败: {e}")

# 输入 y，输出拟合曲线的 x 值
def query_x_from_y(y_input, fit_choice, x_data, y_data):
    try:
        x_fit, y_fit, label = perform_fitting(fit_choice, x_data, y_data)
        # 查找 y_input 对应的 x 值
        if min(y_fit) <= y_input <= max(y_fit):
            x_output = np.interp(y_input, y_fit, x_fit)  # 插值计算 x 值
            print(f"拟合模型: {label}")
            print(f"输入 y = {y_input:.4f}, 输出 x = {x_output:.4f}")
        else:
            print("输入的 y 超出了拟合曲线的范围。")
    except Exception as e:
        print(f"查询失败: {e}")

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取用户指定的 txt 文件
file_name = input("请输入数据文件的名称（包含扩展名，如 data.txt）：").strip()

# 从文件中读取数据
with open(file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 解析文件内容
plot_title = lines[0].split(':')[1].strip()  # 实验标题

# 使用正则表达式提取 x 和 y 的标签
x_label_match = re.search(r':(.+)', lines[1])
x_label = x_label_match.group(1).strip() if x_label_match else "未知 x 轴"

# 初始化存储多组数据的列表
data_groups = []
current_group = {}

# 解析多组数据
for line in lines[2:]:
    line = line.strip()  # 去除行首和行尾的空格
    if not line:  # 跳过空行
        continue
    if line.startswith("y轴名称"):
        if current_group:  # 如果当前组有数据，保存到 data_groups
            if "x_data" not in current_group or "y_data" not in current_group:
                raise ValueError("数据组缺少 x轴数据 或 y轴数据，请检查文件格式。")
            data_groups.append(current_group)
        current_group = {"y_label": line.split(':')[1].strip()}
    elif line.startswith("x轴数据"):
        current_group["x_data"] = np.array(list(map(float, line.split(':')[1].strip().split())))
    elif line.startswith("y轴数据"):
        current_group["y_data"] = np.array(list(map(float, line.split(':')[1].strip().split())))
    else:
        print(f"警告：无法识别的行格式：{line}，请检查文件内容。")

# 保存最后一组数据
if current_group:
    if "x_data" not in current_group or "y_data" not in current_group:
        raise ValueError("数据组缺少 x轴数据 或 y轴数据，请检查文件格式。")
    data_groups.append(current_group)

if not data_groups:
    raise ValueError("未找到有效的数据组，请检查文件格式。")

# 如果只有一组数据，直接处理
if len(data_groups) == 1:
    group = data_groups[0]
    x_data = group["x_data"]
    y_data = group["y_data"]
    y_label = group["y_label"]

# 提供选择方式选项
print("请选择操作模式：")
print("1. 用户选择拟合方式")
print("2. 计算机自动选择拟合方式")
mode_choice = input("请输入选项编号（1 或 2）：")

# 定义拟合模型

# 线性拟合
def linear_model(x, a, b):
    return a * x + b

# 二次拟合
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

# 指数拟合模型
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# 对数拟合模型
def logarithmic_model(x, a, b):
    return a * np.log(b * x)

# 幂函数拟合模型
def power_model(x, a, b):
    return a * x**b

# 傅里叶级数拟合
def fourier_model(x, *params):
    n = len(params) // 2
    a = params[:n]
    b = params[n:]
    result = a[0] + sum(a[i] * np.cos((i + 1) * x) + b[i] * np.sin((i + 1) * x) for i in range(n - 1))
    return result

# 三角函数拟合模型（正弦形式）
def sine_model(x, a, b, c):
    return a * np.sin(b * x + c)

# 高阶多项式拟合
def high_order_polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# 拉普拉斯平滑拟合（示例：简单的平滑处理）
def laplace_smoothing(x, y, alpha=0.1):
    smoothed_y = np.zeros_like(y)
    smoothed_y[0] = y[0]
    for i in range(1, len(y)):
        smoothed_y[i] = alpha * y[i] + (1 - alpha) * smoothed_y[i - 1]
    return smoothed_y

# 计算 R^2（决定系数）
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 封装拟合逻辑为模块
def perform_fitting(fit_choice, x_data, y_data):
    """
    根据拟合选择进行拟合，并返回拟合结果。
    
    参数:
        fit_choice (str): 用户选择的拟合方式
        x_data (array-like): 横坐标数据
        y_data (array-like): 纵坐标数据
    
    返回:
        x_fit (array-like): 拟合曲线的横坐标
        y_fit (array-like): 拟合曲线的纵坐标
        label (str): 拟合曲线的标签
    """
    if fit_choice == "1":
        params, covariance = curve_fit(linear_model, x_data, y_data)
        a, b = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = linear_model(x_fit, a, b)
        label = f'线性拟合: y = {a:.4f}x + {b:.4f}'
    elif fit_choice == "2":
        params, covariance = curve_fit(quadratic_model, x_data, y_data)
        a, b, c = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = quadratic_model(x_fit, a, b, c)
        label = f'二次函数拟合: y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}'
    elif fit_choice == "3":
        params, covariance = curve_fit(exponential_model, x_data, y_data)
        a, b = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = exponential_model(x_fit, a, b)
        label = f'指数拟合: y = {a:.4f} * exp({b:.4f} * x)'
    elif fit_choice == "4":  # 对数拟合
        params, covariance = curve_fit(logarithmic_model, x_data, y_data)
        a, b = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = logarithmic_model(x_fit, a, b)
        label = f'对数拟合: y = {a:.4f} * log({b:.4f} * x)'
    elif fit_choice == "5":  # 幂函数拟合
        params, covariance = curve_fit(power_model, x_data, y_data)
        a, b = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = power_model(x_fit, a, b)
        label = f'幂函数拟合: y = {a:.4f} * x^{b:.4f}'
    elif fit_choice == "6":
        params, covariance = curve_fit(fourier_model, x_data, y_data, p0=[1] * 6)
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = fourier_model(x_fit, *params)
        label = "傅里叶级数拟合"
    elif fit_choice == "7":  # 三角函数拟合
        params, covariance = curve_fit(sine_model, x_data, y_data, p0=[1, 1, 0])
        a, b, c = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = sine_model(x_fit, a, b, c)
        label = f'三角函数拟合: y = {a:.4f} * sin({b:.4f} * x + {c:.4f})'
    elif fit_choice == "8":
        # 动态选择多项式阶数
        degree = min(len(x_data) - 1, 5)  # 阶数不能超过数据点数 - 1
        params = np.polyfit(x_data, y_data, deg=degree)
        poly = np.poly1d(params)
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = poly(x_fit)
        # 构造拟合函数的表达式
        terms = [f"{coef:.4f}x^{i}" if i > 0 else f"{coef:.4f}" 
                 for i, coef in enumerate(params[::-1])]
        equation = " + ".join(terms).replace("x^1", "x")
        label = f"高阶多项式拟合: y = {equation}"
    elif fit_choice == "9":  # 拉普拉斯平滑拟合
        y_fit = laplace_smoothing(x_data, y_data)
        x_fit = x_data  # 平滑拟合不改变 x 数据
        label = '拉普拉斯平滑拟合'
    else:
        raise ValueError("该拟合方法尚未实现或不支持。")
    
    return x_fit, y_fit, label

# 用户选择拟合方式
if mode_choice == "1":
    print("请选择拟合方式：")
    print("1. 线性拟合 (y = ax + b)")
    print("2. 二次拟合 (y = ax^2 + bx + c)")
    print("3. 指数拟合 (y = a * exp(b * x))")
    print("4. 对数拟合 (y = a * log(b * x))")
    print("5. 幂函数拟合 (y = a * x^b)")
    print("6. 傅里叶拟合")
    print("7. 三角函数拟合 (y = a * sin(b * x + c))")
    print("8. 高阶多项式拟合")
    print("9. 拉普拉斯平滑拟合")
    fit_choice = input("请输入拟合方式编号：")
    
    # 绘制多条拟合曲线
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()  # 创建绘图窗口

    # 遍历每组数据并绘制拟合曲线
    for i, group in enumerate(data_groups):
        x_data = group["x_data"]
        y_data = group["y_data"]
        y_label = group["y_label"]

        # 执行拟合
        x_fit, y_fit, label = perform_fitting(fit_choice, x_data, y_data)

        # 计算 R²
        r2 = calculate_r2(y_data, np.interp(x_data, x_fit, y_fit))
        print(f"数据组 {i + 1} - 拟合模型: {label}，R^2 = {r2:.4f}")

        # 绘制数据点和拟合曲线
        ax.scatter(x_data, y_data, label=f'数据组 {i + 1}: {y_label}', alpha=0.7)
        ax.plot(x_fit, y_fit, label=f'拟合组 {i + 1}: {label} (R^2={r2:.4f})')

    # 设置图例和标题
    ax.set_xlabel(x_label)
    ax.set_ylabel("y轴")
    ax.set_title(plot_title)
    ax.legend()
    ax.grid()
    plt.show(block=False)  # 非阻塞显示图像

    # 循环询问用户是否需要查询
    while True:
        print("\n查询选项：")
        print("1. 输入 x 查询拟合曲线的 y 值")
        print("2. 输入 y 查询拟合曲线的 x 值")
        print("3. 退出查询")
        query_choice = input("请输入选项编号（1, 2 或 3）：")
    
        if query_choice in ["1", "2"]:
            # 显示所有拟合曲线供用户选择
            print("\n可用拟合曲线：")
            for i, fit in enumerate(all_fits):
                print(f"{i + 1}. {fit['label']} (R^2={fit['r2']:.4f})")
            curve_choice = int(input("请选择拟合曲线编号：")) - 1

            if 0 <= curve_choice < len(all_fits):
                selected_fit = all_fits[curve_choice]
                x_fit = selected_fit["x_fit"]
                y_fit = selected_fit["y_fit"]
                label = selected_fit["label"]

                if query_choice == "1":
                    x_input = float(input("请输入 x 的值："))
                    if min(x_fit) <= x_input <= max(x_fit):
                        y_output = np.interp(x_input, x_fit, y_fit)  # 插值计算 y 值
                        print(f"拟合模型: {label}")
                        print(f"输入 x = {x_input:.4f}, 输出 y = {y_output:.4f}")
                    else:
                        print("输入的 x 超出了拟合曲线的范围。")
                elif query_choice == "2":
                    y_input = float(input("请输入 y 的值："))
                    if min(y_fit) <= y_input <= max(y_fit):
                        x_output = np.interp(y_input, y_fit, x_fit)  # 插值计算 x 值
                        print(f"拟合模型: {label}")
                        print(f"输入 y = {y_input:.4f}, 输出 x = {x_output:.4f}")
                    else:
                        print("输入的 y 超出了拟合曲线的范围。")
            else:
                print("无效的曲线编号，请重新选择。")
        elif query_choice == "3":
            print("退出查询。")
            break
        else:
            print("无效的选项，请重新输入。")

    plt.ioff()  # 关闭交互模式
    plt.close(fig)  # 关闭图像窗口

# 自动选择拟合方式
elif mode_choice == "2":
    models_with_dashed_lines = ["1", "2", "5"]  # 线性、二次、幂函数拟合
    best_r2 = -np.inf
    best_fit = None
    all_fits = []  # 用于存储所有拟合曲线的结果

    # 询问用户是否绘制虚线
    draw_dashed_lines = input("是否绘制虚线表示其他拟合模型？(y/n)：").strip().lower() == "y"

    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()  # 创建绘图窗口
    ax.scatter(x_data, y_data, label='数据', color='red')  # 绘制原始数据

    for fit_choice in models_with_dashed_lines + ["6", "7", "8", "9"]:
        try:
            x_fit, y_fit, label = perform_fitting(fit_choice, x_data, y_data)
            r2 = calculate_r2(y_data, np.interp(x_data, x_fit, y_fit))

            # 保存拟合结果
            all_fits.append({"fit_choice": fit_choice, "x_fit": x_fit, "y_fit": y_fit, "label": label, "r2": r2})

            # 更新最佳拟合模型
            if r2 > best_r2:
                best_r2 = r2
                best_fit = (x_fit, y_fit, label)

            # 根据用户选择决定是否绘制虚线
            if draw_dashed_lines and fit_choice in models_with_dashed_lines:
                ax.plot(x_fit, y_fit, linestyle='--', label=f'{label} (R^2={r2:.4f})', alpha=0.5)
        except Exception as e:
            print(f"拟合方式 {fit_choice} 出错: {e}")
            continue

    if best_fit:
        x_fit, y_fit, label = best_fit
        print(f"最佳拟合模型: {label}，R^2 = {best_r2:.4f}")
        ax.plot(x_fit, y_fit, label=f'{label} (R^2={best_r2:.4f})', color='blue', linewidth=2)
    else:
        print("未找到合适的拟合模型。")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend()
    ax.grid()

    # 绑定鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show(block=False)  # 非阻塞显示图像

    # 循环询问用户是否需要查询
    while True:
        print("\n查询选项：")
        print("1. 输入 x 查询拟合曲线的 y 值")
        print("2. 输入 y 查询拟合曲线的 x 值")
        print("3. 退出查询")
        query_choice = input("请输入选项编号（1, 2 或 3）：")
    
        if query_choice in ["1", "2"]:
            # 显示所有拟合曲线供用户选择
            print("\n可用拟合曲线：")
            for i, fit in enumerate(all_fits):
                print(f"{i + 1}. {fit['label']} (R^2={fit['r2']:.4f})")
            curve_choice = int(input("请选择拟合曲线编号：")) - 1

            if 0 <= curve_choice < len(all_fits):
                selected_fit = all_fits[curve_choice]
                x_fit = selected_fit["x_fit"]
                y_fit = selected_fit["y_fit"]
                label = selected_fit["label"]

                if query_choice == "1":
                    x_input = float(input("请输入 x 的值："))
                    if min(x_fit) <= x_input <= max(x_fit):
                        y_output = np.interp(x_input, x_fit, y_fit)  # 插值计算 y 值
                        print(f"拟合模型: {label}")
                        print(f"输入 x = {x_input:.4f}, 输出 y = {y_output:.4f}")
                    else:
                        print("输入的 x 超出了拟合曲线的范围。")
                elif query_choice == "2":
                    y_input = float(input("请输入 y 的值："))
                    if min(y_fit) <= y_input <= max(y_fit):
                        x_output = np.interp(y_input, y_fit, x_fit)  # 插值计算 x 值
                        print(f"拟合模型: {label}")
                        print(f"输入 y = {y_input:.4f}, 输出 x = {x_output:.4f}")
                    else:
                        print("输入的 y 超出了拟合曲线的范围。")
            else:
                print("无效的曲线编号，请重新选择。")
        elif query_choice == "3":
            print("退出查询。")
            break
        else:
            print("无效的选项，请重新输入。")

    plt.ioff()  # 关闭交互模式
    plt.close(fig)  # 关闭图像窗口

else:
    print("无效的选项编号，请重新运行程序并输入 1 或 2。")