# STUB for CESM. Needs proper writing and testing.
from Model import Model,type_status
import pathlib
import typing
import os
import logging
import fileinput
import re
import json
from ModelBaseClass import register_param
from namelist_var import NamelistVar

my_logger=logging.getLogger(f"OPTLIM.{__name__}")
print(my_logger,"my_logger")

class GAMIL3_new(Model):

    # def __init__(self,name: str,
    #              reference: pathlib.Path,
    #              post_process: typing.Optional[dict] = None,
    #              model_dir: pathlib.Path = pathlib.Path.cwd(),
    #              config_path: typing.Optional[pathlib.Path] = None,
    #              status: type_status = "CREATED",
    #              parameters: typing.Optional[dict] = None,
    #              engine: typing.Optional["abstractEngine"] = None,
    #              run_info: typing.Optional[dict] = None,
    #              study: typing.Optional["Study"] = None):
    #
    #
    #     # no parameters should be provided unless create or update provided
    #
    #     # call superclass init
    #     super().__init__(name,
    #                                  reference=reference,
    #                                  model_dir=model_dir,
    #                                  config_path=config_path,
    #                                  parameters=parameters,
    #                                  post_process=post_process,
    #                                  study=study,
    #                                  engine=engine,
    #                                  run_info=run_info)

    def __init__(self, name: str, reference: pathlib.Path, **kwargs):
        """"
        gamil3_new Init -- calls super().__init__(*args,**kwargs)
        then sets submit_script to "case.submit" and continue script to case_con.submit
        See Model for documentation on key word parameters
        :param name -- name of the model.
        :param reference -- where reference config lives.
        """
        super().__init__(name, reference, **kwargs)  # call super class init and then override

        self.submit_script=pathlib.Path('case.submit')
        self.continue_script = pathlib.Path('case_con.submit')


    def create_model(self):  # creat_clone？？？！！！需要看看！！！
        """
        Do what ever is necessary. clone the reference case is the best way to proceed.
        :return:
        """

        cmd = ["/BIGDATA2/sysu_atmos_wjliang_1/FG3/scripts/create_clone", "-case", "%s" % (self.model_dir),
               "-clone", self.reference]
        self.run_cmd(cmd)
        submit_script = self.model_dir / self.submit_script
        with open(submit_script, 'wt') as fp:
            print(f"""#!/usr/bin/env bash
    echo "Hello world. Submitting job..."
    cd {self.model_dir}
    sh envset.sh

    file=./user_nl_gamil
    search="pertlim"

    if [ -f "$file" ]; then
        sed -i "/$search/s/^/!/" "$file"
    else
        echo "$file 不存在"
    fi

    ./{self.name}.build
    ./{self.name}.submit
    echo "Submitted job"
    """, file=fp)
        submit_script.chmod(0o777)

        continue_script = self.model_dir / self.continue_script
        with open(continue_script, 'wt') as fp:
            print(f"""#!/usr/bin/env bash
    echo "Hello world. Submitting job..."
    cd {self.model_dir}

    #directory="/BIGDATA2/sysu_atmos_wjliang_1/FG3/run/{self.name}/run"

    #latest_file=$(find "$directory" -type f -name "*gamil.h0*" | sort -n | tail -1)
    #year_month=$(echo "$latest_file" | grep -oE "[0-9]{4}-[0-9]{2}")

    #target_year_month="2011-12"

    #latest_date=$(date -d "$year_month-01" +%s)
    #target_date=$(date -d "$target_year_month-01" +%s)
    #months_diff=$(( ($latest_date - $target_date) / 60 / 60 / 24 / 30 ))

    #files11=$(find "$directory" -type f -name "*gamil.h0*")
    #file_count=$(echo "$files11" | wc -l)

    #if [ $file_count -eq 0 ]; then   

        #file="./user_nl_gamil"
        #if [ -f "$file" ]; then
        #    sed -i 's/!\(pertlim = 1e-14\)/\1/g' "$file"
        #else
        #    echo "文件 $file 不存在"
        #fi
    sh envset.sh

    file=./user_nl_gamil
    search="pertlim"

    if [ -f "$file" ]; then
        sed -i "/$search/s/^/!/" "$file"
    else
        echo "$file 不存在"
    fi

    ./{self.name}.build
    ./{self.name}.submit

    #file=./user_nl_gamil
    #search="pertlim"
    
    #if [ -f "$file" ]; then
    #    sed -i "/$search/s/^/!/" "$file"
    #else
    #    echo "$file 不存在"
    #fi

    #./{self.name}.build
    #./{self.name}.submit

    #else
    #    ./xmlchange CONTINUE_RUN=TRUE
    #    ./xmlchange STOP_N=$((-months_diff))
    #    echo "Changed STOP_N to:"
    #    echo $((-months_diff))
    #    ./{self.name}.submit
    #fi


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

        os.system("cd %s && ./cesm_setup" % (self.model_dir))

        # cmd = ["%s/cesm_setup"%(self.model_dir)]
        # self.run_cmd(cmd)
        # my_logger.debug(f"run {cmd}")

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

        # newname = self.model_dir/str(self.model_dir)[80:]
        # str(genname) + '/' + str(genname)[80:]  # str(genname)[80:]#str(genname)+'/'+str(genname)[80:]
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
        print(self.model_dir, "self.model_dir1")
        # print(self.model_dir / str(self.model_dir)[83:],"self.model_dir2")
        pth = self.model_dir / (self.name + ".run")  # pth of model script
        pth_archive = self.model_dir / "Tools/st_archive.sh"
        modifystr = '## modified'
        with fileinput.input(pth, inplace=True, backup='.bak') as f:  # 没修改的话在run文件第一行插入一句修改的标记话
            for line in f:
                if re.search(modifystr, line):
                    raise ValueError(f"Already modified Script {pth}")
                elif re.match("^yhrun -n", line):  # f.isfirstline():  # first line
                    print(f"{self.set_status_script} {self.config_path} RUNNING {modifystr}")
                    print(line[0:-1])  # print line out.
                elif line.startswith("grep 'SUCCESSFUL TERMINATION' $CPLLogFile  ||"):
                    print(
                        f"grep 'SUCCESSFUL TERMINATION' $CPLLogFile  ||{self.set_status_script} {self.config_path} FAILED {modifystr}")
                    print(line[0:-1])
                else:
                    print(line[0:-1])  # print out the original line.

        with fileinput.input(pth_archive, inplace=True, backup='.bak') as f:  # 在archive文件最后一行插入一句完成了的标记话
            for line in f:
                if re.search(modifystr, line):
                    raise ValueError(f"Already modified archive {pth_archive}")
                elif 'short-term archiving completed successfully' in line:  # re.match('short-term archiving completed successfully',line):
                    print(f"{self.set_status_script} {self.config_path} SUCCEEDED {modifystr}")
                    print(line[0:-1])  # print out the original line.
                else:
                    print(line[0:-1])  # print out the original line.

    @register_param("ensemble_member")
    def ensemble_member(self, value: typing.Optional[int]):

        if value == 0:
            return None
        nl = NamelistVar(type_name='namelist_var',nl_var="pertlim", namelist="atmexp", filepath=pathlib.Path("user_nl_gamil"))
        return (nl, value * 1e-14)


pth = pathlib.Path(__file__).parent /'parameter_config/GAMIL3_Parameters.csv'
GAMIL3_new.update_from_file(pth, duplicate=True)

