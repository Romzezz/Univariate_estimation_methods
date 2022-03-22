#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML
import sympy as sp
import numpy as np
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
import time
import scipy.optimize
import plotly.graph_objects as go
from sympy.utilities.lambdify import implemented_function
warnings.filterwarnings('ignore')


# In[2]:


def ask_input(ask_bounds=False, ask_initial_point=False):
    '''
    Запрашивает у пользователя входные данные.
            Параметры:
                    ask_bounds (bool, default=False): 
                        Если True, запращивает у пользователя границы аргумента, точность метода
                        и макс. кол-во итераций
                    ask_initial_point (bool, default=False): 
                        Если True, запрашивает у пользователя начальную точку, первый и второй
                        параметр для условия Вольфе, а также максимальное ограничение по аргументу
                        
            Возвращаемое значение:
                    result (dict):
                        словарь, значениями которого являются введённые пользователем значения
    '''
    func_str = input('Введите функцию в аналитическом виде: ')
    func = sp.sympify(func_str)
    arg = list(func.free_symbols)[0]
    func = sp.lambdify(arg, func)
    result = {'func': func}
    result['func_str'] = func_str
    if ask_bounds:
        bounds = input('Введите ограничения по аргументу: ')
        bounds = tuple(map(float, map(sp.sympify, bounds.split())))
        result['bounds'] = bounds
        accuracy = input('Введите точность алгоритма (оставьте пустым для значения по умолчанию): ')
        accuracy = None if not accuracy else float(accuracy)
        result['accuracy'] = accuracy
        max_iter = input('Введите макс. кол-во итераций (оставьте пустым для значения по умолчанию): ')
        max_iter = None if not max_iter else int(max_iter)
        result['max_iter'] = max_iter
    if ask_initial_point:
        initial_point = float(input('Введите начальную точку (для метода BFGS): '))
        result['initial_point'] = initial_point
        if result.get('accuracy', -1) == -1:
            accuracy = input('Введите точность алгоритма (оставьте пустым для значения по умолчанию): ')
            accuracy = None if not accuracy else float(accuracy)
            result['accuracy'] = accuracy
            max_iter = input('Введите макс. кол-во итераций (оставьте пустым для значения по умолчанию): ')
            max_iter = None if not max_iter else int(accuracy)
            result['max_iter'] = max_iter
        c1 = input('Введите параметр для первого условия Вольфе (оставьте пустым для значения по умолчанию): ')
        c1 = None if not c1 else float(c1)
        result['c1'] = c1
        c2 = input('Введите параметр для второго условия Вольфе (оставьте пустым для значения по умолчанию): ')
        c2 = None if not c2 else float(c2)
        result['c2'] = c2
        max_arg = input('Введите максимальное ограничение по аргументу (оставьте пустым для значения по умолчанию): ')
        max_arg = None if not max_arg else float(max_arg)
        result['max_arg'] = max_arg
    return result


def _find_center(func, x1, x2, x3):
    '''
    Находит центр параболы, построенной по трём точкам.
            Параметры:
                    func (function): 
                        Исследуемая функция
                    x1 (float): 
                        Первая точка
                    x2 (float):
                        Вторая точка
                    x3 (float):
                        Третья точка
                        
            Возвращаемое значение:
                    center (float):
                        Координата центра параболы
    '''
    if (x1, func(x1)) == (x2, func(x2)) or (x2, func(x2)) == (x3, func(x3)) or (x1, func(x1)) == (x3, func(x3)):
        return
    f_1, f_2, f_3 = func(x1), func(x2), func(x3)
    a1 = (f_2 - f_1) / (x2 - x1)
    a2 = 1/(x3 - x2)*((f_3 - f_1)/(x3 - x1) - (f_2 - f_1)/(x2 - x1))
    center = 0.5*(x1 + x2 - a1/a2)
    return center

def _show_convergency(data):
    '''
    Строит график сходимости алгоритма и выводит на экран размеры интервалов.
            Параметры:
                    data (list): 
                        Список значений размеров интервалов
                        
            Возвращаемое значение:
                    None
    '''
    print(data)
    plt.figure()
    plt.plot(list(range(len(data))), data)
    plt.xlabel('iter num')
    plt.ylabel('interval size')
    plt.title('convergency estimation')
    plt.show()


# In[3]:


def golden_ratio(func, bounds, accuracy=None, max_iter=None, show_interim_results=False, show_convergency=False, return_data=False):
    '''
    Находит минимум функции методом золотого сечения на заданном интервале.
            Параметры:
                    func (function): 
                        Исследуемая функция
                    bounds (tuple or list):
                        Исследуемый интервал
                    accuracy (float, default=None):
                        Точность метода
                    max_iter (int, default=None):
                        Максимальное количество итераций
                    show_interim_results (bool, default=False):
                        Если True, выводит на экран датасет с промежуточными результатами
                    show_convergency (bool, default=False):
                        Если True, выводит на экран график сходимости алгоритма
                    return_data (bool, default=False):
                        Если True, добавляет в возвращаяемый словарь датасет с промежуточными результатами
                        
            Возвращаемое значение:
                    result (dict):
                        Словарь, значениями которого являются точка минимума функции, значение функции в этой точке и флаг
                    
    '''
    if accuracy == None:
        accuracy = 10e-5
    if max_iter == None:
        max_iter = 500
    iter_num = 0
    left, right = bounds
    ratio = (np.sqrt(5) - 1) / 2    
    interim_results = pd.DataFrame(columns=['N', 'a', 'b', 'x', 'y', 'f(x)', 'f(y)'])
    
    while abs(right - left) > accuracy:
        if iter_num >= max_iter:
            break
            
        x = right - ratio*(right - left)
        y = left + ratio*(right - left)
        
        row = {
            'N': iter_num,
            'a': left,
            'b': right,
            'x': x,
            'y': y,
            'f(x)': func(x),
            'f(y)': func(y)
        }
        interim_results = interim_results.append(row, ignore_index=True)

        if func(x) < func(y):
            right = y
        else:
            left = x
        
        iter_num += 1
    
    flag = 1 if iter_num >= max_iter else 0
    x_min = (left + right) / 2
    f_min = func(x_min)
    
    interim_results.set_index('N', inplace=True)
    if show_interim_results:
        display(HTML(interim_results.to_html()))
    if show_convergency:
        interval_sizes = list(interim_results['b'] - interim_results['a'])
        _show_convergency(interval_sizes)
    result = {'arg': x_min, 'func': f_min, 'flag': flag}
    if return_data:
        result['data'] = interim_results
    return result


# In[4]:


def parabola_method(func, bounds, accuracy=None, max_iter=None, show_interim_results=False, show_convergency=False, return_data=False):
    '''
    Находит минимум функции парабол на заданном интервале.
            Параметры:
                    func (function): 
                        Исследуемая функция
                    bounds (tuple or list):
                        Исследуемый интервал
                    accuracy (float, default=None):
                        Точность метода
                    max_iter (int, default=None):
                        Максимальное количество итераций
                    show_interim_results (bool, default=False):
                        Если True, выводит на экран датасет с промежуточными результатами
                    show_convergency (bool, default=False):
                        Если True, выводит на экран график сходимости алгоритма
                    return_data (bool, default=False):
                        Если True, добавляет в возвращаяемый словарь датасет с промежуточными результатами
                        
            Возвращаемое значение:
                    result (dict):
                        Словарь, значениями которого являются точка минимума функции, значение функции в этой точке и флаг
                    
    '''
    if accuracy == None:
        accuracy = 10e-5
    if max_iter == None:
        max_iter = 500
    iter_num = 0
    left, right = bounds
    x1, x2, x3 = left, (left + right) / 2, right
    prev_min = (left + right) / 2
    curr_min = _find_center(func, x1, x2, x3)
    interim_results = pd.DataFrame(columns=['N', 'x1', 'x2', 'x3', 'u', 'f(u)'])
    
    while abs(curr_min - prev_min) > accuracy:
        if iter_num >= max_iter:
            break
            
        c = min(x2, curr_min)
        d = max(x2, curr_min)
        
        row = {
            'N': iter_num,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'u': curr_min,
            'f(u)': func(curr_min)
        }
        interim_results = interim_results.append(row, ignore_index=True)
        
        if func(c) < func(d):
            x1, x2, x3 = x1, c, d
        else:
            x1, x2, x3 = c, d, right
            
        prev_min = curr_min
        curr_min = _find_center(func, x1, x2, x3)
            
        iter_num += 1

    flag = 1 if iter_num >= max_iter else 0
    interim_results.set_index('N', inplace=True)
    if show_interim_results:
        display(HTML(interim_results.to_html()))
    if show_convergency:
        interval_sizes = list(interim_results['x3'] - interim_results['x1'])
        _show_convergency(interval_sizes)
    result = {'arg': curr_min, 'func': func(curr_min), 'flag': flag}
    if return_data:
        result['data'] = interim_results
    return result


# In[5]:


def combined_brent(func, bounds, accuracy=None, max_iter=None, show_interim_results=False, show_convergency=False, return_data=False):
    '''
    Находит минимум функции комбинированным методом Брента на заданном интервале.
            Параметры:
                    func (function): 
                        Исследуемая функция
                    bounds (tuple or list):
                        Исследуемый интервал
                    accuracy (float, default=None):
                        Точность метода
                    max_iter (int, default=None):
                        Максимальное количество итераций
                    show_interim_results (bool, default=False):
                        Если True, выводит на экран датасет с промежуточными результатами
                    show_convergency (bool, default=False):
                        Если True, выводит на экран график сходимости алгоритма
                    return_data (bool, default=False):
                        Если True, добавляет в возвращаяемый словарь датасет с промежуточными результатами
                        
            Возвращаемое значение:
                    result (dict):
                        Словарь, значениями которого являются точка минимума функции, значение функции в этой точке и флаг
                    
    '''
    if accuracy == None:
        accuracy = 10e-5
    if max_iter == None:
        max_iter = 500
    iter_num = 0
    ratio = (3 - np.sqrt(5)) / 2
    
    left, right = bounds
    x_min = w = v = left + ratio*(right - left)
    d_curr = d_prev = right - left
    
    interim_results = pd.DataFrame(columns=['a', 'b', 'x', 'w', 'v', 'u'])
    
    while max(x_min - left, right - x_min) > accuracy:
        if iter_num >= max_iter:
            break
            
        g = d_prev / 2
        d_prev = d_curr
        u = _find_center(func, x_min, w, v)
        if not u or (u < left or u > right) or abs(u - x_min) > g:
            if x_min < (left + right) / 2:
                u = x_min + ratio*(right - x_min)
                d_prev = right - x_min
            else:
                u = x_min - ratio*(x_min - left)
                d_prev = (x_min - left)
        d_curr = abs(u - x_min)
        
        row = {
            'N': iter_num,
            'a': left,
            'b': right,
            'x': x_min,
            'w': w,
            'v': v,
            'u': u
        }
        interim_results = interim_results.append(row, ignore_index=True)
        
        if func(u) > func(x_min):
            if u < x_min:
                left = u
            else:
                right = u
            if func(u) <= func(w) or w == x_min:
                v = w
                w = u
            else:
                if func(u) <= func(v) or v == x_min or v == w:
                    v = u
        else:
            if u < x_min:
                right = x_min
            else:
                left = x_min
            v = w
            w = x_min
            x_min = u
            
        iter_num += 1
        
    flag = 1 if iter_num >= max_iter else 0
    interim_results.set_index('N', inplace=True)
    if show_interim_results:
        display(HTML(interim_results.to_html()))
    if show_convergency:
        interval_sizes = list(interim_results['b'] - interim_results['a'])
        _show_convergency(interval_sizes)
    result = {'arg': x_min, 'func': func(x_min), 'flag': flag}
    if return_data:
        result['data'] = interim_results
    return result


# In[6]:


def bfgs_method(func, x0, c1=None, c2=None, max_iter=None, max_arg=None, accuracy=None, show_interim_results=False, return_data=False):
    '''
    Находит минимум функции методом BFGS.
            Параметры:
                    func (function): 
                        Исследуемая функция
                    x0 (float):
                        Начальная точка
                    c1 (float, default=None):
                        Первый параметр условия Вольфе
                    c2 (float, default=None):
                        Второй параметр условия Вольфе
                    max_iter (int, default=None):
                        Максимальное количество итераций
                    max_arg (float, default=None):
                        Ограничение на максимальное значение аргумента
                    accuracy (float, default=None):
                        Точность метода
                    show_interim_results (bool, default=False):
                        Если True, выводит на экран датасет с промежуточными результатами
                    return_data (bool, default=False):
                        Если True, добавляет в возвращаяемый словарь датасет с промежуточными результатами
                        
            Возвращаемое значение:
                    result (dict):
                        Словарь, значениями которого являются точка минимума функции, значение функции в этой точке и флаг
                    
    '''
    if accuracy == None:
        accuracy = 10e-5
    if max_iter == None:
        max_iter = 500
    if c1 == None:
        c1 = 10e-4
    if c2 == None:
        c2 = 0.1
    if max_arg == None:
        max_arg = 100
    func = sp.sympify(func)
    arg = list(func.free_symbols)[0]
    der = sp.lambdify(arg, sp.diff(func, arg))
    func = sp.lambdify(arg, func)

    k = 0
    gfk = der(x0)
    I = 1
    Hk = I
    xk = x0
    error_occured = False
    interim_results = pd.DataFrame(columns=['N', 'H', 'x_curr', 'x_next', 'p', 's', 'y'])
    flag = None
   
    while np.linalg.norm(gfk) > accuracy:
        if k >= max_iter:
            break
            
        pk = -np.dot(Hk, gfk)

        line_search = scipy.optimize.line_search(func, der, xk, pk, c1=c1, c2=c2)
        alpha_k = line_search[0]
        
        if alpha_k == None:
            error_occured = True
            break
        
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        
        row = {
            'N': k,
            'H': Hk,
            'x_curr': xk,
            'p': pk,
            's': sk,
        }
        
        xk = xkp1
        row['x_next'] = xkp1
        
        gfkp1 = der(xkp1)
        yk = gfkp1 - gfk
        row['y'] = yk
        gfk = gfkp1
        
        interim_results = interim_results.append(row, ignore_index=True)
        
        if xkp1 > max_arg:
            flag = 3
            break
        
        k += 1
        
        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk * yk
        A2 = I - ro * yk * sk
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk *sk)
        
    flag = 2 if k >= max_iter else flag
    flag = 4 if error_occured else flag
    if flag == None:
        flag = 0
    interim_results.set_index('N', inplace=True)
    if show_interim_results:
        display(HTML(interim_results.to_html()))
    result = {'arg': xk, 'func': func(xk), 'flag': flag}
    if return_data:
        result['data'] = interim_results
    return result


# In[7]:


def solve(compare=False):
    if compare:
        data = ask_input(1, 1)
        values = [['Полученное решение', 'Время выполнения (ms)', 'Количество итераций']]
        for algorithm in (golden_ratio, parabola_method, combined_brent, bfgs_method):
            start = time.time()
            if algorithm != bfgs_method:
                result = algorithm(
                    data['func'], data['bounds'], accuracy=data['accuracy'], max_iter=data['max_iter'],
                    return_data=True
                )
            else:
                result = algorithm(
                    data['func_str'], data['initial_point'], accuracy=data['accuracy'],
                    max_iter=data['max_iter'], max_arg=data['max_arg'], c1=data['c1'], c2=data['c2'],
                    return_data=True
                )
            if result['flag'] in (0, 1):
                solution = result['arg']
            else:
                solution = 'error occured'
            end = time.time()
            duration = end - start
            iter_num = len(result['data'].index)
            values.append([solution, duration, iter_num])
        acc_time_start = time.time()
        x1, x2 = data['bounds']
        acc_solution = scipy.optimize.fminbound(data['func'], x1, x2)
        acc_time_end = time.time()
        acc_duration = acc_time_end - acc_time_start
        values.append([acc_solution, acc_duration, '-'])
        fig = go.Figure(data=[go.Table(header=dict(values=['Параметр', 'метод золотого сечения', 'метод парабол', 'комбинированный метод Брента', 'BFGS', 'Оптимальный точный алгоритм']),
                 cells=dict(values=values))])
        fig.show()
        return
        
    else:
        method = int(input(
        """
        Выберите метод решения:
        1 - метод золотого сечения
        2 - метод парабол
        3 - комбинированный метод Брента
        4 - BFGS
        Метод: 
        """
        ))
        name_to_algo = {
            1: golden_ratio,
            2: parabola_method,
            3: combined_brent
        }
        show_interim_results = bool(int(input('Показать промежуточные результаты? 1-да / 0-нет: ')))
        if method in (1, 2, 3):
            show_convergency = bool(int(input('Показать график сходимости? 1-да / 0-нет: ')))
            data = ask_input(1, 0)
            return name_to_algo[method](
                data['func'], data['bounds'], accuracy=data['accuracy'], max_iter=data['max_iter'],
                show_interim_results=show_interim_results, show_convergency=show_convergency
            )
        else:
            data = ask_input(0, 1)
            return bfgs_method(
                data['func_str'], data['initial_point'], accuracy=data['accuracy'],
                max_iter=data['max_iter'], max_arg=data['max_arg'], c1=data['c1'], c2=data['c2'],
                return_data=True, show_interim_results=show_interim_results
            )


# In[ ]:





# In[ ]:




