package com.udea.bancoudea.service;

import com.udea.bancoudea.dto.TransactionDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.entity.Transaction;
import com.udea.bancoudea.repository.CustomerRepository;
import com.udea.bancoudea.repository.TransactionRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class TransactionServiceTest {

    @Mock
    private TransactionRepository transactionRepository;

    @Mock
    private CustomerRepository customerRepository;

    @InjectMocks
    private TransactionService transactionService;

    private Customer sender;
    private Customer receiver;
    private TransactionDTO transactionDTO;

    @BeforeEach
    void setUp() {
        sender = new Customer(1L, "ACC-001", "Juan", "Perez", 1000.0);
        receiver = new Customer(2L, "ACC-002", "Maria", "Lopez", 500.0);

        transactionDTO = new TransactionDTO();
        transactionDTO.setSenderAccountNumber("ACC-001");
        transactionDTO.setReceiverAccountNumber("ACC-002");
        transactionDTO.setAmount(200.0);
    }

    // --- transferMoney ---

    @Test
    void transferMoney_realizaTransferenciaCorrectamente() {
        Transaction savedTransaction = new Transaction(1L, "ACC-001", "ACC-002", 200.0, LocalDateTime.now());

        when(customerRepository.findByAccountNumber("ACC-001")).thenReturn(Optional.of(sender));
        when(customerRepository.findByAccountNumber("ACC-002")).thenReturn(Optional.of(receiver));
        when(transactionRepository.save(any(Transaction.class))).thenReturn(savedTransaction);

        TransactionDTO result = transactionService.transferMoney(transactionDTO);

        assertNotNull(result);
        assertEquals("ACC-001", result.getSenderAccountNumber());
        assertEquals("ACC-002", result.getReceiverAccountNumber());
        assertEquals(200.0, result.getAmount());

        // Verificar que los saldos se actualizaron
        assertEquals(800.0, sender.getBalance());
        assertEquals(700.0, receiver.getBalance());

        verify(customerRepository).save(sender);
        verify(customerRepository).save(receiver);
        verify(transactionRepository).save(any(Transaction.class));
    }

    @Test
    void transferMoney_lanzaExcepcionSiSaldoInsuficiente() {
        transactionDTO.setAmount(5000.0); // mas que el saldo del emisor

        when(customerRepository.findByAccountNumber("ACC-001")).thenReturn(Optional.of(sender));
        when(customerRepository.findByAccountNumber("ACC-002")).thenReturn(Optional.of(receiver));

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> transactionService.transferMoney(transactionDTO));

        assertEquals("Sender Balance not enough", ex.getMessage());
        verify(transactionRepository, never()).save(any());
    }

    @Test
    void transferMoney_lanzaExcepcionSiCuentaEmisorNoExiste() {
        when(customerRepository.findByAccountNumber("ACC-001")).thenReturn(Optional.empty());

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> transactionService.transferMoney(transactionDTO));

        assertEquals("Sender Account Number not found", ex.getMessage());
    }

    @Test
    void transferMoney_lanzaExcepcionSiCuentaReceptorNoExiste() {
        when(customerRepository.findByAccountNumber("ACC-001")).thenReturn(Optional.of(sender));
        when(customerRepository.findByAccountNumber("ACC-002")).thenReturn(Optional.empty());

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> transactionService.transferMoney(transactionDTO));

        assertEquals("Receiver Account Number not found", ex.getMessage());
    }

    @Test
    void transferMoney_lanzaExcepcionSiSenderNulo() {
        TransactionDTO dto = new TransactionDTO();
        dto.setAmount(100.0);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> transactionService.transferMoney(dto));

        assertEquals("Sender Account Number cannot be null", ex.getMessage());
    }

    @Test
    void transferMoney_lanzaExcepcionSiReceiverNulo() {
        TransactionDTO dto = new TransactionDTO();
        dto.setSenderAccountNumber("ACC-001");
        dto.setAmount(100.0);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> transactionService.transferMoney(dto));

        assertEquals("Receiver Account Number cannot be null", ex.getMessage());
    }

    // --- getTransactionsForAccount ---

    @Test
    void getTransactionsForAccount_retornaTransacciones() {
        Transaction t = new Transaction(1L, "ACC-001", "ACC-002", 200.0, LocalDateTime.now());

        when(transactionRepository.findBySenderAccountNumberOrReceiverAccountNumber("ACC-001", "ACC-001"))
                .thenReturn(List.of(t));

        List<TransactionDTO> result = transactionService.getTransactionsForAccount("ACC-001");

        assertEquals(1, result.size());
        assertEquals("ACC-001", result.get(0).getSenderAccountNumber());
        assertEquals(200.0, result.get(0).getAmount());
    }

    @Test
    void getTransactionsForAccount_retornaListaVaciaSiNoHayTransacciones() {
        when(transactionRepository.findBySenderAccountNumberOrReceiverAccountNumber("ACC-999", "ACC-999"))
                .thenReturn(List.of());

        List<TransactionDTO> result = transactionService.getTransactionsForAccount("ACC-999");

        assertTrue(result.isEmpty());
    }
}
