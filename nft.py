#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import pandas as pd
import numpy as np
import time
import os
import random
from progressbar import *
from tqdm import tqdm, trange
# 引入config 配置文件
from config import CONFIG
#忽略警告消息
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
文件整体实现功能：
首先确定图层文件的路径是否有问题，然后告诉我们总共有多少种可能性的NFT，
根据给定的稀有度要求，按照对应的稀有度概率要求，随机生成所有的标签组合，
最后按照一定的顺序，组合成所有的图片集合，并且除去相同的图片，
把所有的标签生成csv文件记录下来

函数：
    parse_config() 解析配置文件并确保它有效，确定文件夹不是空的图层，根据给定的稀有度，生成相应的稀有度数组
    get_weighted_rarities 加权稀有度并返回一个总和为 1 的 numpy 数组
    generate_single_image 给定代表图层的文件路径数组，生成单个图像
    get_total_combinations 计算 获取不同的可能的组合的 总数
    select_index 根据稀有权重选择索引，抽一个随机数，看这个随机数在哪个区间，就选哪个特征的图层
    generate_trait_set_from_config 在给定稀有度的情况下生成一组特征，并且找到所有特征的路径
    generate_images 生成图像集，并对生成的重复图像进行删除
    '''



# 解析配置文件并确保它有效
def parse_config():
    
    # 输入特征必须放在 assets 文件夹中。 如果您想将其命名为其他名称，请更改此值。
    assets_path = 'assets'

    # 循环遍历 CONFIG 中定义的所有图层标签
    for layer in CONFIG:

        # 进入 assets文件夹寻找图层文件夹
        layer_path = os.path.join(assets_path, layer['directory'])
        
        # 按排序顺序获取特征数组
        #traits[]数组记录着所有的图层每个设计风格的图片名字
        traits = sorted([trait for trait in os.listdir(layer_path) if trait[0] != '.'])


        # 如果不是必须的图层，那么就把相应图层的required改成false，就会按照概率生成或者不生成图层，通过在特征数组的开头添加一个 None标记
        if not layer['required']:
            traits = [None] + traits
        
        # 生成最终的稀有 权重 数组
        ''' '''

        if layer['rarity_weights'] is None:
            rarities = [1 for x in traits]  # 没有权重关系，等概率抽一个
        elif layer['rarity_weights'] == 'random':  # 随机权重关系，生成随机概率
            rarities = [random.random() for x in traits]
        elif type(layer['rarity_weights'] == 'list'):
            assert len(traits) == len(layer['rarity_weights']), "Make sure you have the current number of rarity weights"
            rarities = layer['rarity_weights']
        else:
            raise ValueError("Rarity weights is invalid")
        
        rarities = get_weighted_rarities(rarities)
        
        # Re-assign final values to main CONFIG
        layer['rarity_weights'] = rarities
        layer['cum_rarity_weights'] = np.cumsum(rarities)  #  cumsum功能是一个一个的返回数组累计和[1,2,3] -> [1,3,6]
        layer['traits'] = traits


# 加权稀有度并返回一个总和为 1 的 numpy 数组
def get_weighted_rarities(arr):
    return np.array(arr)/ sum(arr)

# 给定代表图层的文件路径数组，生成单个图像
def generate_single_image(filepaths, output_filename=None):
    
    # 将第一层作为背景
    bg = Image.open(os.path.join('assets', filepaths[0])).convert("RGBA")
    
    
    # 遍历第 1 层到第 n 层并将它们堆叠在另一个之上
    for filepath in filepaths[1:]:
        if filepath.endswith('.png'):
            img = Image.open(os.path.join('assets', filepath)).convert("RGBA")
            bg.paste(img, (0,0), img)
    
    # 将最终图像保存到所需位置
    if output_filename is not None:
        bg.save(output_filename)
    else:
        # 如果未指定输出文件名，则使用时间戳命名图像并将其保存在 output/single_images
        if not os.path.exists(os.path.join('output', 'single_images')):
            os.makedirs(os.path.join('output', 'single_images'))
        bg.save(os.path.join('output', 'single_images', str(int(time.time())) + '.png'))


# Generate a single image with all possible traits
# generate_single_image(['Background/green.png', 
#                        'Body/brown.png', 
#                        'Expressions/standard.png',
#                        'Head Gear/std_crown.png',
#                        'Shirt/blue_dot.png',
#                        'Misc/pokeball.png',
#                        'Hands/standard.png',
#                        'Wristband/yellow.png'])


# 获取不同的可能组合的总数
def get_total_combinations():
    total = 1
    for layer in CONFIG:
        total = total * len(layer['traits'])
    return total


# 根据稀有权重选择索引
'''cum_rarities这是一个从0--1的数组，里面的每个数字其实代表了一个区间,比如[0，0.2，0.4，0.6，0.8，1 ]，那么再生成一个随机数
0.5，那么它应该是在0.4-0.6区间，那么返回0.4对应的i也就是2，但是因为增加了0，其实这个index对应的是第三个区间c也就是0.4-0.6
a(0,0.2)    b(0.2,0.4)    c(0.4,0.6)
正好是0.6这个标签'''
def select_index(cum_rarities, rand):
    print(cum_rarities)
    cum_rarities = [0] + list(cum_rarities)
    for i in range(len(cum_rarities) - 1):
        if rand >= cum_rarities[i] and rand <= cum_rarities[i+1]:
            return i
    # Should not reach here if everything works okay
    return None


# 在给定稀有度的情况下生成一组特征
def generate_trait_set_from_config():
    
    trait_set = []
    trait_paths = []
    
    for layer in CONFIG:
        # Extract list of traits and cumulative rarity weights  提取特征列表和累积稀有权重
        traits, cum_rarities = layer['traits'], layer['cum_rarity_weights']

        # Generate a random number 生成一个随机数
        rand_num = random.random()

        # Select an element index based on random number and cumulative rarity weights
        # 根据随机数和累积稀有权重选择元素索引
        idx = select_index(cum_rarities, rand_num)

        # 将所选特征添加到特征集
        trait_set.append(traits[idx])

        # 如果已选择特征，则将特征路径添加到特征路径 traits全是标签图片的名字
        if traits[idx] is not None:
            trait_path = os.path.join(layer['directory'], traits[idx])
            trait_paths.append(trait_path)
        
    return trait_set, trait_paths


# Generate the image set. Don't change drop_dup 生成图像集，并对生成的重复图像进行删除
def generate_images(edition, count, drop_dup=True):
    
    # 初始化一个空的稀有度表
    rarity_table = {}
    for layer in CONFIG:
        rarity_table[layer['name']] = []

    # 定义 {edition_num} 的输出路径
    op_path = os.path.join('output', 'edition ' + str(edition), 'images')

    # 要求将最终图像命名为 000、001、...
    zfill_count = len(str(count - 1))
    
    # 如果输出目录不存在，则创建它
    if not os.path.exists(op_path):
        os.makedirs(op_path)

    # 创建图像
    for n in trange(count):


        # 设置图片名称  zfill返回指定长度的字符串，原字符串右对齐，前面填充0。
        image_name = str(n).zfill(zfill_count) + '.png'
        
        # Get a random set of valid traits based on rarity weights
        trait_sets, trait_paths = generate_trait_set_from_config()

        # 生成实际图像
        generate_single_image(trait_paths, os.path.join(op_path, image_name))
        
        # Populate the rarity table with metadata of newly created image
        # 使用新创建图像的元数据填充稀有表
        for idx, trait in enumerate(trait_sets):
            if trait is not None:
                rarity_table[CONFIG[idx]['name']].append(trait[: -1 * len('.png')])
            else:
                rarity_table[CONFIG[idx]['name']].append('none')
    # Create the final rarity table by removing duplicate creat
    # 通过删除重复的创建最终的稀有度表
    rarity_table = pd.DataFrame(rarity_table).drop_duplicates()
    print("Generated %i images, %i are distinct" % (count, rarity_table.shape[0]))
    
    if drop_dup:
        # 获取重复图像列表
        img_tb_removed = sorted(list(set(range(count)) - set(rarity_table.index)))

        #删除重复的图像
        print("Removing %i images..." % (len(img_tb_removed)))

        #op_path = os.path.join('output', 'edition ' + str(edition))
        for i in img_tb_removed:
            os.remove(os.path.join(op_path, str(i).zfill(zfill_count) + '.png'))

        # 重命名图像，使其按顺序编号
        for idx, img in enumerate(sorted(os.listdir(op_path))):
            os.rename(os.path.join(op_path, img), os.path.join(op_path, str(idx).zfill(zfill_count) + '.png'))
    
    
    # 修改稀有度表以反映移除情况
    rarity_table = rarity_table.reset_index()
    rarity_table = rarity_table.drop('index', axis=1)
    return rarity_table

# Main function. Point of entry
def main():

    print("Checking assets...")
    parse_config()
    print("Assets look great! We are good to go!")
    print()

    tot_comb = get_total_combinations()
    print("You can create a total of %i distinct avatars" % (tot_comb))
    print()

    print("How many avatars would you like to create? Enter a number greater than 0: ")
    while True:
        num_avatars = int(input())
        if num_avatars > 0:
            break
    
    print("What would you like to call this edition?: ")
    edition_name = input()

    print("Starting task...")
    rt = generate_images(edition_name, num_avatars)

    print("Saving metadata...")
    rt.to_csv(os.path.join('output', 'edition ' + str(edition_name), 'metadata.csv'))

    print("Task complete!")


# Run the main function
main()