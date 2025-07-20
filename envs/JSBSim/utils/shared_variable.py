class GlobalVars:
    shared_acmi_id=None #acmi前置文件号
    shared_missile_shootpoint = True #导弹事件，当为true代表导弹在此刻被发射，之后下一个时刻改为false。为了给数据统计使用
    use_autoshoot_in_render = False #如果要使用初始化时自动发射导弹模型，需要将render_2v2中的use_autoshoot_in_render改为True