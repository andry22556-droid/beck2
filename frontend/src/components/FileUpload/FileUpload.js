import React, {useRef, useState} from 'react';
import {Button, CircularProgress, FormControl, FormControlLabel, FormLabel, Radio, RadioGroup} from "@mui/material";
import useStyles from "./styles";
import axios from "axios";

const FileUpload = (props) => {

    const classes = useStyles();
    const fileInputRef = useRef(null)
    const [filename, setFilename] = useState(null)
    const [platform, setPlatform] = useState(null)
    const [platformTg, setPlatformTg] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const formData = new FormData()

    function transformData(response) {
        const transformedList = [];

        for (const key in response) {
            if (typeof response[key] === "object") {
                const sublist = [key, ...Object.values(response[key])];
                transformedList.push(sublist);
            } else {
                transformedList.push([key, response[key]]);
            }
        }
        return transformedList;
    }

    function handleClick() {
        let name = fileInputRef.current.files[0].name
        setIsLoading(true)
        formData.append("file", fileInputRef.current.files[0])
        if(platformTg==="TGStat" || platformTg==="livedune"){
            axios.post(`/${platform}/file/?platform=${platformTg}`, formData,
                {headers: {"Content-Type": "multipart/form-data"}}).then(res => {
                props.setData(transformData(res.data))
                props.setStateScreen(1)
            })
        }
        else {
            axios.post(`/${platform}/file`, formData,
                {headers: {"Content-Type": "multipart/form-data"}}).then(res => {
                props.setData(transformData(res.data))
                props.setStateScreen(1)
            })
        }
    }

    function handleChangeType(event){
        setPlatform(event.target.value)
    }

    function handleChangeTgPlatform(event){
        setPlatformTg(event.target.value)
    }

    return (
        <div className={classes.container}>
            <p className={classes.title}>Сбор аналитических данных блога</p>
            <RadioGroup style={{marginTop: "2vh", marginBottom: "2vh"}} onClick={(e) => handleChangeType(e)}>
                <FormControl>
                    <FormLabel style={{color: "#E6E6E6"}}>Выберите платформу</FormLabel>
                    <RadioGroup
                        row
                    >
                        <FormControlLabel style={{color: "#00F43A"}} value="vk"
                                          control={<Radio style={{color: "#00F43A"}}/>} label="ВКонтакте"/>
                        <FormControlLabel style={{color: "#00F43A"}} value="tg"
                                          control={<Radio style={{color: "#00F43A"}}/>} label="Telegram"/>
                        <FormControlLabel style={{color: "#00F43A"}} value="yt"
                                          control={<Radio style={{color: "#00F43A"}}/>} label="YouTube"/>
                        <FormControlLabel style={{color: "#00F43A"}} value="zn"
                                          control={<Radio style={{color: "#00F43A"}}/>} label="Dzen"/>
                    </RadioGroup>
                </FormControl>
            </RadioGroup>
            {!!platform && platform==="tg" &&
                <RadioGroup style={{marginTop: "2vh", marginBottom: "2vh"}}
                            onClick={(e) => handleChangeTgPlatform(e)}>
                    <FormControl>
                        <FormLabel style={{color: "#E6E6E6"}}>Выберите сервис</FormLabel>
                        <RadioGroup
                            row
                        >
                            <FormControlLabel style={{color: "#00F43A"}} value="TGStat"
                                              control={<Radio style={{color: "#00F43A"}}/>} label="TGStat"/>
                            <FormControlLabel style={{color: "#00F43A"}} value="livedune"
                                              control={<Radio style={{color: "#00F43A"}}/>} label="LIVEDUNE"/>
                        </RadioGroup>
                    </FormControl>
                </RadioGroup>
            }
            {((!!platform && platform!=="tg") || (!!platform && platform==="tg" && !!platformTg))&&
                <div>
                    <div className={classes.sendContainer}>
                        <div className={classes.fileNameContainer}>{filename || "выберите файл для отправки"}</div>
                        <Button
                            component="label"
                            style={openButton}
                        >
                            Обзор
                            <input type="file" ref={fileInputRef} hidden onChange={(e) => {
                                if (!!e.target.files[0]) {
                                    setFilename(e.target.files[0].name)
                                }
                            }}
                            />
                        </Button>
                    </div>
                    <div style={{display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", marginTop: "5vh", gap: "2vw"}}>
                        <Button onClick={handleClick} style={sendButton}>
                            Обработать изображения
                        </Button>
                        {!!isLoading &&
                            <CircularProgress style={{color: "#00F43A"}}></CircularProgress>
                        }
                    </div>
                </div>
            }
        </div>
    );
};

export default FileUpload;


let openButton = {
    background: "#00F43A",
    borderRadius: "20px",
    fontFamily: 'Nunito',
    fontStyle: "normal",
    fontWeight: 400,
    fontSize: "24px",
    lineHeight: "33px",
    color: "#181818",
    marginLeft: "10px",
    height: "8vh",
    width: "8vw"
}

let sendButton = {
    background: "#39393A",
    borderRadius: "20px",
    fontFamily: 'Nunito',
    fontsStyle: "normal",
    fontWeight: 400,
    fontSize: "24px",
    color: "#00F43A",
    width: "24vw",
    lineHeight: "33px",
    height: "8vh"
}
