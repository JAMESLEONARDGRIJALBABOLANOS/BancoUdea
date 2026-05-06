package com.udea.bancoudea.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.udea.bancoudea.dto.TransactionDTO;
import com.udea.bancoudea.service.TransactionService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(TransactionController.class)
class TransactionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private TransactionService transactionService;

    private TransactionDTO transactionDTO;

    @BeforeEach
    void setUp() {
        transactionDTO = new TransactionDTO(1L, "ACC-001", "ACC-002", 200.0, LocalDateTime.now());
    }

    @Test
    void transfer_retornaTransaccionExitosa() throws Exception {
        when(transactionService.transferMoney(any(TransactionDTO.class))).thenReturn(transactionDTO);

        mockMvc.perform(post("/api/transactions/transfer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(transactionDTO)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.senderAccountNumber").value("ACC-001"))
                .andExpect(jsonPath("$.amount").value(200.0));
    }

    @Test
    void transfer_retorna400CuandoSaldoInsuficiente() throws Exception {
        when(transactionService.transferMoney(any(TransactionDTO.class)))
                .thenThrow(new IllegalArgumentException("Sender Balance not enough"));

        mockMvc.perform(post("/api/transactions/transfer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(transactionDTO)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.message").value("Sender Balance not enough"));
    }

    @Test
    void transfer_retorna400CuandoCuentaNoExiste() throws Exception {
        when(transactionService.transferMoney(any(TransactionDTO.class)))
                .thenThrow(new IllegalArgumentException("Sender Account Number not found"));

        mockMvc.perform(post("/api/transactions/transfer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(transactionDTO)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.message").value("Sender Account Number not found"));
    }

    @Test
    void getTransactions_retornaHistorial() throws Exception {
        when(transactionService.getTransactionsForAccount("ACC-001")).thenReturn(List.of(transactionDTO));

        mockMvc.perform(get("/api/transactions/ACC-001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].senderAccountNumber").value("ACC-001"))
                .andExpect(jsonPath("$[0].amount").value(200.0));
    }

    @Test
    void getTransactions_retornaListaVaciaSiNoHayHistorial() throws Exception {
        when(transactionService.getTransactionsForAccount("ACC-999")).thenReturn(List.of());

        mockMvc.perform(get("/api/transactions/ACC-999"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isEmpty());
    }
}
