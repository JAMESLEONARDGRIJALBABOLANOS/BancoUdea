package com.udea.bancoudea.service;

import com.udea.bancoudea.dto.TransactionDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.entity.Transaction;
import com.udea.bancoudea.mapper.TransactionMapper;
import com.udea.bancoudea.repository.CustomerRepository;
import com.udea.bancoudea.repository.TransactionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TransactionService {

    @Autowired
    private TransactionRepository transactionRepository;

    @Autowired
    private CustomerRepository customerRepository;

    private final TransactionMapper transactionMapper = TransactionMapper.INSTANCE;

    public TransactionDTO transferMoney(TransactionDTO transactionDTO) {
        validateAccountNumbers(transactionDTO);

        Customer sender = customerRepository.findByAccountNumber(transactionDTO.getSenderAccountNumber())
                .orElseThrow(() -> new IllegalArgumentException("Sender Account Number not found"));
        Customer receiver = customerRepository.findByAccountNumber(transactionDTO.getReceiverAccountNumber())
                .orElseThrow(() -> new IllegalArgumentException("Receiver Account Number not found"));

        validateBalance(sender, transactionDTO.getAmount());

        sender.setBalance(sender.getBalance() - transactionDTO.getAmount());
        receiver.setBalance(receiver.getBalance() + transactionDTO.getAmount());
        customerRepository.save(sender);
        customerRepository.save(receiver);

        Transaction transaction = new Transaction();
        transaction.setSenderAccountNumber(sender.getAccountNumber());
        transaction.setReceiverAccountNumber(receiver.getAccountNumber());
        transaction.setAmount(transactionDTO.getAmount());
        transaction.setTimestamp(java.time.LocalDateTime.now());
        return transactionMapper.toDTO(transactionRepository.save(transaction));
    }

    private void validateAccountNumbers(TransactionDTO dto) {
        if (dto.getSenderAccountNumber() == null) {
            throw new IllegalArgumentException("Sender Account Number cannot be null");
        }
        if (dto.getReceiverAccountNumber() == null) {
            throw new IllegalArgumentException("Receiver Account Number cannot be null");
        }
    }

    private void validateBalance(Customer sender, Double amount) {
        if (sender.getBalance() < amount) {
            throw new IllegalArgumentException("Sender Balance not enough");
        }
    }

    public List<TransactionDTO> getTransactionsForAccount(String accountNumber) {
        List<Transaction> transactions = transactionRepository.findBySenderAccountNumberOrReceiverAccountNumber(accountNumber,accountNumber);
        return transactions.stream().map(transactionMapper::toDTO).toList();
    }

}
