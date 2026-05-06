package com.udea.bancoudea.cucumber.steps;

import com.udea.bancoudea.dto.TransactionDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.entity.Transaction;
import com.udea.bancoudea.repository.CustomerRepository;
import com.udea.bancoudea.repository.TransactionRepository;
import com.udea.bancoudea.service.TransactionService;
import io.cucumber.java.After;
import io.cucumber.java.en.*;
import org.springframework.beans.factory.annotation.Autowired;

import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TransactionSteps {

    @Autowired
    private TransactionService transactionService;

    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private TransactionRepository transactionRepository;

    private Customer sender;
    private Customer receiver;
    private TransactionDTO resultDTO;
    private Exception thrownException;
    private List<TransactionDTO> transactionList;

    @After
    public void limpiar() {
        transactionRepository.deleteAll();
        if (sender != null) customerRepository.findByAccountNumber(sender.getAccountNumber())
                .ifPresent(customerRepository::delete);
        if (receiver != null) customerRepository.findByAccountNumber(receiver.getAccountNumber())
                .ifPresent(customerRepository::delete);
        sender = null;
        receiver = null;
        thrownException = null;
    }

    @Given("una cuenta emisora {string} con saldo {double}")
    public void unaCuentaEmisoraConSaldo(String cuenta, Double saldo) {
        customerRepository.findByAccountNumber(cuenta).ifPresent(customerRepository::delete);
        sender = customerRepository.save(new Customer(null, cuenta, "Emisor", "Test", saldo));
    }

    @Given("una cuenta receptora {string} con saldo {double}")
    public void unaCuentaReceptoraConSaldo(String cuenta, Double saldo) {
        customerRepository.findByAccountNumber(cuenta).ifPresent(customerRepository::delete);
        receiver = customerRepository.save(new Customer(null, cuenta, "Receptor", "Test", saldo));
    }

    @When("se transfiere {double} de {string} a {string}")
    public void seTransfiere(Double monto, String cuentaEmisor, String cuentaReceptor) {
        try {
            TransactionDTO dto = new TransactionDTO();
            dto.setSenderAccountNumber(cuentaEmisor);
            dto.setReceiverAccountNumber(cuentaReceptor);
            dto.setAmount(monto);
            resultDTO = transactionService.transferMoney(dto);
        } catch (Exception e) {
            thrownException = e;
        }
    }

    @Then("el saldo de {string} es {double}")
    public void elSaldoDeCuentaEs(String cuenta, Double saldoEsperado) {
        Customer c = customerRepository.findByAccountNumber(cuenta)
                .orElseThrow(() -> new RuntimeException("Cuenta no encontrada: " + cuenta));
        assertEquals(saldoEsperado, c.getBalance());
    }

    @Then("la transaccion queda registrada")
    public void laTransaccionQuedaRegistrada() {
        assertNotNull(resultDTO);
        assertNotNull(resultDTO.getId());
    }

    @Then("se lanza una excepcion de transferencia con mensaje {string}")
    public void seLanzaUnaExcepcionDeTransferenciaConMensaje(String mensaje) {
        assertNotNull(thrownException);
        assertEquals(mensaje, thrownException.getMessage());
    }

    @Given("que la cuenta {string} no existe")
    public void queLaCuentaNoExiste(String cuenta) {
        customerRepository.findByAccountNumber(cuenta).ifPresent(customerRepository::delete);
    }

    @When("se intenta transferir {double} desde {string} a {string}")
    public void seIntentaTransferir(Double monto, String cuentaEmisor, String cuentaReceptor) {
        try {
            TransactionDTO dto = new TransactionDTO();
            dto.setSenderAccountNumber(cuentaEmisor);
            dto.setReceiverAccountNumber(cuentaReceptor);
            dto.setAmount(monto);
            transactionService.transferMoney(dto);
        } catch (Exception e) {
            thrownException = e;
        }
    }

    @Given("una cuenta {string} con transacciones registradas")
    public void unaCuentaConTransaccionesRegistradas(String cuenta) {
        customerRepository.findByAccountNumber(cuenta).ifPresent(customerRepository::delete);
        customerRepository.findByAccountNumber("ACC-DEST").ifPresent(customerRepository::delete);

        sender = customerRepository.save(new Customer(null, cuenta, "Test", "User", 1000.0));
        receiver = customerRepository.save(new Customer(null, "ACC-DEST", "Dest", "User", 500.0));

        Transaction t = new Transaction(null, cuenta, "ACC-DEST", 100.0, LocalDateTime.now());
        transactionRepository.save(t);
    }

    @When("se consulta el historial de {string}")
    public void seConsultaElHistorialDe(String cuenta) {
        transactionList = transactionService.getTransactionsForAccount(cuenta);
    }

    @Then("se retorna al menos una transaccion")
    public void seRetornaAlMenosUnaTransaccion() {
        assertNotNull(transactionList);
        assertFalse(transactionList.isEmpty());
    }
}
