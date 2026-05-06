package com.udea.bancoudea.controller;

import com.udea.bancoudea.dto.TransactionDTO;
import com.udea.bancoudea.service.TransactionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/transactions")
public class TransactionController {

    private final TransactionService transactionService;

    public TransactionController(TransactionService transactionService) {
        this.transactionService = transactionService;
    }

    // Realizar una transferencia entre cuentas
    @PostMapping("/transfer")
    public ResponseEntity<TransactionDTO> transfer(@RequestBody TransactionDTO transactionDTO) {
        return ResponseEntity.ok(transactionService.transferMoney(transactionDTO));
    }

    // Obtener historial de transacciones de una cuenta
    @GetMapping("/{accountNumber}")
    public ResponseEntity<List<TransactionDTO>> getTransactions(@PathVariable String accountNumber) {
        return ResponseEntity.ok(transactionService.getTransactionsForAccount(accountNumber));
    }
}
