import React, {useEffect, useState} from 'react';
import {Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow} from "@mui/material";

const TableCsv = (props) => {

    const [head, setHead] = useState([])
    const [rows, setRows] = useState([])

    useEffect(() => {
        let new_data = props.data
        setRows(new_data)
        if(new_data[0][1][2]) setHead(["Изображение", "Подписчики", "Просмотры"])
        else setHead(["Изображение", "Целевой показатель"])

    }, [props.data])

    return (
        <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label="simple table">
                <TableBody>
                    <TableRow
                        sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                        <TableCell align={"center"} style={{color: "#FFFFFF", background: "#272727"}}>{head[0]}</TableCell>
                        <TableCell align={"center"} style={{color: "#FFFFFF",background: "#272727"}} >{head[1]}</TableCell>
                        <TableCell align={"center"} style={{color: "#FFFFFF",background: "#272727"}} >{head[2] && head[2]}</TableCell>
                    </TableRow>
                    {props.data.map((row, index) => (
                        <TableRow
                            key={row.name}
                            sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                        >
                            <TableCell align={"center"} style={{color: "#FFFFFF", background: "#272727"}}>{row[1][0]}</TableCell>
                            <TableCell align={"center"} style={{color: "#FFFFFF",background: "#272727"}} >{Number(row[1][1]).toFixed(2)}</TableCell>
                            <TableCell align={"center"} style={{color: "#FFFFFF",background: "#272727"}} >{row[1][2] && Number(row[1][2]).toFixed(2)}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default TableCsv;

let cellStyle = {
    background: "#272727",
}
