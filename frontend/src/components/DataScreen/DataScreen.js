import React, { useEffect, useState } from 'react';
import useStyles from "./styles";
import { Button } from "@mui/material";
import iconBack from "../../images/IconBack.svg";
import { saveAs } from 'file-saver';
import TableCsv from "../TableCsv/TableCsv";

const DataScreen = (props) => {

    const classes = useStyles();
    const [content, setContent] = useState(null);

    useEffect(() => {
        if (props.data !== null) {
            let data = props.data;
            setContent(Object.entries(data));
        }
    }, []);

    function downloadCSV(data) {
        const transformedData = props.data // Подставьте функцию transformData
        const csvContent = arrayToCSV(transformedData);

        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8" });
        saveAs(blob, "processed_data.csv");
    }

    function goToBack() {
        props.setStateScreen(0);
    }

    return (
        <div className={classes.container}>
            <div className={classes.titleContainer}>
                <img onClick={goToBack} src={iconBack} className={classes.imgContainer} />
                <div style={{ display: "flex", flexDirection: "row", justifyContent: "center", alignItems: "center" }}>
                    <Button onClick={() => downloadCSV(content)} style={downloadButton}>
                        Скачать обработанный датасет в CSV
                    </Button>
                </div>
            </div>
            {!!content &&
                <TableCsv data={content} />
            }
        </div>
    );
};

export default DataScreen;

let downloadButton = {
    marginTop: 20,
    marginBottom: 20,
    background: "#00F43A",
    borderRadius: "20px",
    fontFamily: 'Nunito',
    fontStyle: "normal",
    fontWeight: 600,
    fontSize: "16px",
    lineHeight: "33px",
    color: "#181818",
    marginLeft: "15vw",
    height: "6vh"
}

// Преобразование массива данных в строку CSV
function arrayToCSV(data) {
    const csvRows = [];
    for (const row of data) {
        const csvRow = row.map(value => `"${value}"`).join(",");
        csvRows.push(csvRow);
    }
    return csvRows.join("\n");
}
