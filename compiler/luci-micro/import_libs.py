import shutil, re

shutil.copytree('../stdex', './lib/stdex', dirs_exist_ok=True)
shutil.copytree('../oops', './lib/oops', dirs_exist_ok=True)
shutil.copytree('../pepper-str', './lib/pepper-str', dirs_exist_ok=True)
shutil.copytree('../pepper-str', './lib/pepper-str', dirs_exist_ok=True)
shutil.copytree('../logo', './lib/logo', dirs_exist_ok=True)
shutil.copytree('../logo-core', './lib/logo-core', dirs_exist_ok=True)
shutil.copytree('../loco', './lib/loco', dirs_exist_ok=True)
shutil.copytree('../locomotiv', './lib/locomotiv', dirs_exist_ok=True)
shutil.copytree('../angkor', './lib/angkor', dirs_exist_ok=True)

pattern  = r"target_link_libraries\(.+ (nncc_common|nncc_coverage)\)"
for libname in ['locomotiv', 'angkor', 'loco']:
	filename = f'./lib/{libname}/CMakeLists.txt'
	new_file = []
	with open(filename, 'r') as f:
	   lines = f.readlines()
	for line in lines:
	    match = re.search(pattern, line)
	    if not match:
	        new_file.append(line)

	with open(filename, 'w') as f:
	     f.seek(0)
	     f.writelines(new_file)