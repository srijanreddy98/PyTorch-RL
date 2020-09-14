import re
s = ''
with open('./all_files.txt') as utils_filelist:
    con = utils_filelist.readlines()
    filelist = map(lambda x: x.split('\n')[0].split('./')[1], con)
    for filename in filelist:
        with open('./'+filename) as file:
            con = file.read()
            con = re.sub('disable bundler(?s)(.*)enable bundler', '', con)
            s += con
    with open('./examples/ppo_gym_spin.py') as f:
        s += re.sub('disable bundler(?s)(.*)enable bundler', '', f.read())
    with open('./final.py', 'w') as bundle: bundle.write(s)
