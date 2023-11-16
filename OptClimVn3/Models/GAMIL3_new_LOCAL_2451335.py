# STUB for CESM. Needs proper writing and testing.
from Model import Model,type_status
import pathlib
import typing
import os
import logging
import fileinput
import re
import json

my_logger=logging.getLogger(f"OPTLIM.{__name__}")
print(my_logger,"my_logger")

class GAMIL3_new(Model):

    def __init__(self,name: str,
                 reference: pathlib.Path,
                 post_process: typing.Optional[dict] = None,
                 model_dir: pathlib.Path = pathlib.Path.cwd(),
                 config_path: typing.Optional[pathlib.Path] = None,
                 status: type_status = "CREATED",
                 parameters: typing.Optional[dict] = None,
                 engine: typing.Optional["abstractEngine"] = None,
                 run_info: typing.Optional[dict] = None,
                 study: typing.Optional["Study"] = None):


        # no parameters should be provided unless create or update provided

        # call superclass init
        super().__init__(name,
                                     reference=reference,
                                     model_dir=model_dir,
                                     config_path=config_path,
                                     parameters=parameters,
                                     post_process=post_process,
                                     study=study,
                                     engine=engine,
                                     run_info=run_info)

        self.submit_script=pathlib.Path('case.submit')
        self.continue_script = pathlib.Path('case_con.submit')
        #model_dir=self.model_dir
        #self.params_gamil = result_dict

        # self.create_model()
        # self.modify_model()             #在两个文件中插入两句标记话
        # self.set_params(parameters)
        # self.set_status('INSTANTIATED')
        # #self.set_status('INSTANTIATED')
        # cmd=self.submit_cmd()
        # self.run_cmd(cmd)



    # CESM does not set namelists just write variables to a directory.
    # Creation/submission then  uses this information to build a case
    # largely uses Model methods.
    def read_CESM_user_vars(self,path: pathlib) -> dict:
        """
        :param path: path where user params are
        :return: dict of parameters and value
        """
        result = dict()
        # Check if the file exists. If not, return an empty dict.
        if not path.exists():
            return result
        lines = path.read_text().splitlines()
        for line in lines:
            # Check if the line starts with '&' or '/'
            if line.startswith('&') or line.startswith('/'):
                # print(line)
                # If it does, use the whole line as the variable name and set the value to None or an empty string.
                var = line.strip()
                # print(var)
                value = None  # You can set it to an empty string ("") if you prefer.
            else:
                # If the line doesn't start with '&' or '/', proceed with the original parsing logic.
                var, value = line.split("=")
                var = var.strip()
                value = value.strip()
                if value.startswith("'") or value.startswith('"'):
                    value = str(value[1:-1])
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Unable to parse as int or float, keep it as a string.
            result[var] = value
        return result

    def write_CESM_vars(self,parameters: dict, path: pathlib.Path) -> None:
        """
        :param parameters: dict of parameters to write out
        :return:
        """
        # generate text to write and then write it out.
        with open(path, "wt") as fp:
            for param, value in parameters.items():
                if value is None:
                    fp.write(param + "\n")  # print(fp, param)  # Just print the parameter name
                elif isinstance(value, str):
                    if param[0:]=="Nudge_Model" or param[0:]=="Nudge_Hwin_Invert" or param[0:]=="Nudge_Vwin_Invert":
                        output = value#"'" + value + "'"
                        fp.write(f"{param} = {output}\n")  # print(fp, f"{param} = {output}")
                        if param == "Nudge_Vwin_Invert":
                            fp.write(f"/\n")  # print(fp, f"{param} = {output}")
                    # elif (param=="START_TIME" or param=="RUN_TARGET" or "ensembleMember" == param):
                    #     pass
                    else:
                        output = "'" + value + "'"
                        fp.write(f"{param} = {output}\n")  # print(fp, f"{param} = {output}")
                        if param == "Nudge_Vwin_Invert":
                            fp.write(f"/\n")  # print(fp, f"{param} = {output}")
                elif isinstance(value, (int, float)):  # Check if value is int or float
                    output = f"{value}"
                    fp.write(f"{param} = {output}\n")  # print(fp, f"{param} = {output}")
                    if param == "dthdpmn_chg":
                        fp.write(f"/\n")  # print(fp, f"{param} = {output}")
                else:
                    raise ValueError(f"Do not know how to deal with type {type(value)}")

    def set_params(self,parameters:typing.Optional[dict]=None) -> None:  # parameters来自他的算法得到的新的参数集
        """

        :param parameters: dict of parameters to set. Will override existing parameters
        :return: Nothing
        """
        if parameters is None:
            parameters = self.parameters
        ens = parameters.pop("ensembleMember",0)
        namelist_path = self.model_dir/"user_nl_gamil"
        exist_params = self.read_CESM_user_vars(namelist_path)

        # result_dict = {}
        # count = 0
        # for key, value in parameters.items():
        #     if count < 10:
        #         result_dict[key] = value
        #         count += 1
        #     else:
        #         break


        # exist_params.update(parameters)
        #params=self.params_gamil
        print(parameters,"params_new")
        exist_params.update(parameters)
        # for key, value in parameters.items():
        #     exist_params[key] = value
        self.write_CESM_vars(exist_params,namelist_path)

    def replace_strings_in_script(self,script_file, replacement1, replacement2):
        try:
            # 打开脚本文件以进行读取
            with open(script_file, 'r') as file:
                script_contents = file.read()

            # 使用replace方法替换字符串
            modified_script = script_contents.replace("./new1", f"{replacement1}")
            modified_script = modified_script.replace("./new", f"./{replacement2}")

            # 打开脚本文件以进行写入，将修改后的内容写入文件
            with open(script_file, 'w') as file:
                file.write(modified_script)

            print(
                f'Successfully replaced "./new1" with "./{replacement1}" and "./new" with "./{replacement2}" in {script_file}.')
        except Exception as e:
            print(f'Error: {str(e)}')

    def create_model(self):  # creat_clone？？？！！！需要看看！！！
        """
        Do what ever is necessary. clone the reference case is the best way to proceed.
        :return:
        """

        cmd = ["/BIGDATA2/sysu_atmos_wjliang_1/FG3/scripts/create_clone","-case", "%s"%(self.model_dir),"-clone", self.reference]
        self.run_cmd(cmd)
        submit_script=self.model_dir/self.submit_script
        with open(submit_script,'wt') as fp:
            print(f"""#!/usr/bin/env bash
echo "Hello world. Submitting job..."
cd {self.model_dir}
sh envset.sh
./{self.name}.build
./{self.name}.submit
echo "Submitted job"
""",file=fp)
        submit_script.chmod(0o777)

        continue_script = self.model_dir / self.continue_script
        with open(continue_script, 'wt') as fp:
            print(f"""#!/usr/bin/env bash
        echo "Hello world. Submitting job..."
        cd {self.model_dir}
        sh envset.sh
        ./xmlchange CONTINUE_RUN=TRUE
        ./{self.name}.build
        ./{self.name}.submit
        echo "Submitted job"
        """, file=fp)
        continue_script.chmod(0o777)


        # submit_path=self.model_dir/(self.reference.name+".submit").rename("case.submit")
        # my_logger.debug(f"run {cmd}")
        # cmd = ["chmod", "777", "%s/case.submit"%(self.model_dir)]
        # self.run_cmd(cmd)
        # my_logger.debug(f"run {cmd}")
        # cmd = ["chmod", "777", "%s/case_con.submit" % (self.model_dir)]
        # self.run_cmd(cmd)
        # my_logger.debug(f"run {cmd}")

        os.system("cd %s && ./cesm_setup"%(self.model_dir))

        #cmd = ["%s/cesm_setup"%(self.model_dir)]
        #self.run_cmd(cmd)
        #my_logger.debug(f"run {cmd}")

        # cmd = ["%s/cesm_setup" % (self.model_dir)]
        # self.run_cmd(cmd)
        # my_logger.debug(f"run {cmd}")

        # os.system("%s/create_clone -case %s -clone %s" % (
        #     "/BIGDATA2/sysu_atmos_wjliang_1/FG3/scripts", genname,
        #     '/BIGDATA2/sysu_atmos_wjliang_1/FG3/scripts/amip1d_nudging_new'))  # liangwj???

        # os.system("chmod 777 %s/case.submit" % (genname))
        # print(genname, "genname")
        # print("finished change params. sets")

    def submit_cmd(self) -> typing.List[str]:
        """

        :return: cmd (a list of strings) to submit the model   #INSTANTIATED准备提交的代码行
        """

        #newname = self.model_dir/str(self.model_dir)[80:]
        #str(genname) + '/' + str(genname)[80:]  # str(genname)[80:]#str(genname)+'/'+str(genname)[80:]
        # self.replace_strings_in_script(self.model_dir / 'case.submit', str(self.model_dir), self.name)
        # self.replace_strings_in_script(self.model_dir / 'case_con.submit', str(self.model_dir), self.name)
        if self.status in ["INSTANTIATED", "PERTURBED"]:
            script = "case.submit"  # "%s.submit"%(name1)
        elif self.status == "CONTINUE":
            # perhaps just call xmlchange  CONTINUE=TRUE
            # os.system("%s/xmlchange CONTINUE_RUN=TRUE"%(genname))  #liangwj???
            script = "case_con.submit"  # "%s.submit"%(name1)
        else:
            raise ValueError(f"Status {self.status} not expected ")

        runCode = self.run_info.get("runCode")
        runTime = self.run_info.get("runTime")  # NUll -- leave as is
        # use xmlchange to set both of runCode (project) and runTime if not null
        script = self.model_dir / script  # self.model_dir / script  # provide full path. -- which might just be case.build etc. This probably needs changing
        cmd = [str(script)]
        return cmd

    def modify_model(self):  # 修改模式设置
        """
        Make changes to run_simple_model.py
        Adds in cmds to set status.
        :return: nada
        """
        super().modify_model()
        print(self.model_dir,"self.model_dir1")
        #print(self.model_dir / str(self.model_dir)[83:],"self.model_dir2")
        pth = self.model_dir /(self.name+".run") # pth of model script
        pth_archive = self.model_dir/"Tools/st_archive.sh"
        modifystr = '## modified'
        with fileinput.input(pth, inplace=True, backup='.bak') as f:   #没修改的话在run文件第一行插入一句修改的标记话
            for line in f:
                if re.search(modifystr, line):
                    raise ValueError(f"Already modified Script {pth}")
                elif re.match("^yhrun -n",line):#f.isfirstline():  # first line
                    print(f"{self.set_status_script} {self.config_path} RUNNING {modifystr}")
                    print(line[0:-1])  # print line out.
                else:
                    print(line[0:-1])  # print out the original line.

        with fileinput.input(pth_archive, inplace=True, backup='.bak') as f:  #在archive文件最后一行插入一句完成了的标记话
            for line in f:
                if re.search(modifystr, line):
                    raise ValueError(f"Already modified archive {pth_archive}")
                elif 'short-term archiving completed successfully' in line:#re.match('short-term archiving completed successfully',line):
                    print(f"{self.set_status_script} {self.config_path} SUCCEEDED {modifystr}")
                    print(line[0:-1])  # print out the original line.
                else:
                    print(line[0:-1])  # print out the original line.


    # def process(self):
    #     """
    #     Run the post-processing, store output and set status to COMPLETED.
    #     "Contract" for a post-processing script
    #     1) takes a json file as input (arg#1) and puts output in file (arg#2).
    #     2) It is being ran in the model_directory.
    #      arg#1 needs json.load to read the json file. Code should expect a dict and use the postProcess entry.
    #          This allows it ot read in and act on a StudyConfig file.
    #      arg#2 can be .json or .csv or .nc
    #     :return: output from post-processing.
    #     """
    #     status: Model.type_status = 'PROCESSED'
    #     if self.fake:  # faking?
    #         self.set_status(status)  # just update the status
    #         return
    #
    #     input_file = self.model_dir / self._post_process_input  # generate json file to hold post process info
    #     my_logger.debug(f"Dumping post_process to {input_file}")
    #     output = dict(postProcess=self.post_process)  # wrap post process in dict
    #     with open(input_file, 'w') as fp:
    #         json.dump(output, fp)
    #     # dump the post-processing dict for the post-processing to  pick up.
    #
    #     post_process_output = self.model_dir / self._post_process_output
    #     run_dir=self.model_dir.parent.parent/"run"/self.model_dir.name/"atm/hist"
    #     result = self.run_cmd(self.post_process_cmd_script,
    #                           cwd=run_dir)
    #
    #     my_logger.debug(f"runing in {run_dir}")
    #
    #     #"/BIGDATA2/sysu_atmos_wjliang_1/FG3/run/"+str(self.model_dir)[83:]+"/atm/hist")#self.model_dir)  #liangwj
    #
    #     # print(self.model_dir, "self.model_dir_liangwj")
    #     # print("/BIGDATA2/sysu_atmos_wjliang_1/FG3/run/" + str(self.model_dir)[83:] + "/atm/hist", "liangwj_pptest")
    #
    #     # get in the simulated obs which also sets them
    #     self.read_simulated_obs(post_process_output)
    #     self.set_status(status)
    #     return result
