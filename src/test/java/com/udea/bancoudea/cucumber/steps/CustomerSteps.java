package com.udea.bancoudea.cucumber.steps;

import com.udea.bancoudea.dto.CustomerDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.mapper.CustomerMapper;
import com.udea.bancoudea.repository.CustomerRepository;
import com.udea.bancoudea.service.CustomerService;
import io.cucumber.java.en.*;
import org.springframework.beans.factory.annotation.Autowired;

import static org.junit.jupiter.api.Assertions.*;

public class CustomerSteps {

    @Autowired
    private CustomerService customerService;

    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private CustomerMapper customerMapper;

    private CustomerDTO inputDTO;
    private CustomerDTO resultDTO;
    private Exception thrownException;

    @Given("un cliente con nombre {string}, apellido {string}, cuenta {string} y saldo {double}")
    public void unClienteConDatos(String nombre, String apellido, String cuenta, Double saldo) {
        inputDTO = new CustomerDTO(null, nombre, apellido, cuenta, saldo);
    }

    @When("se crea el cliente")
    public void seCreaElCliente() {
        // Limpiar por si ya existe
        customerRepository.findByAccountNumber(inputDTO.getAccountNumber())
                .ifPresent(c -> customerRepository.delete(c));
        resultDTO = customerService.createCustomer(inputDTO);
    }

    @Then("el cliente es retornado con los mismos datos")
    public void elClienteEsRetornadoConLosMimosDatos() {
        assertNotNull(resultDTO);
        assertEquals(inputDTO.getFirstName(), resultDTO.getFirstName());
        assertEquals(inputDTO.getLastName(), resultDTO.getLastName());
        assertEquals(inputDTO.getAccountNumber(), resultDTO.getAccountNumber());
        assertEquals(inputDTO.getBalance(), resultDTO.getBalance());
        // Limpiar
        customerRepository.deleteById(resultDTO.getId());
    }

    @Given("un cliente guardado con ID {long}")
    public void unClienteGuardadoConID(Long id) {
        Customer customer = new Customer(null, "ACC-TEST-" + id, "Test", "User", 100.0);
        Customer saved = customerRepository.save(customer);
        // Guardamos el id real asignado en el contexto para usarlo despues
        inputDTO = customerMapper.toDTO(saved);
    }

    @When("se busca el cliente con ID {long}")
    public void seBuscaElClienteConID(Long id) {
        try {
            Long idToUse = (inputDTO != null && inputDTO.getId() != null) ? inputDTO.getId() : id;
            resultDTO = customerService.getCustomerById(idToUse);
        } catch (Exception e) {
            thrownException = e;
        }
    }

    @Then("se retorna el cliente correctamente")
    public void seRetornaElClienteCorrectamente() {
        assertNotNull(resultDTO);
        assertEquals(inputDTO.getId(), resultDTO.getId());
        customerRepository.deleteById(resultDTO.getId());
    }

    @Given("que no existe un cliente con ID {long}")
    public void queNoExisteUnClienteConID(Long id) {
        inputDTO = null;
        customerRepository.findById(id).ifPresent(c -> customerRepository.delete(c));
    }

    @Then("se lanza una excepcion con mensaje {string}")
    public void seLanzaUnaExcepcionConMensaje(String mensaje) {
        assertNotNull(thrownException);
        assertEquals(mensaje, thrownException.getMessage());
        thrownException = null;
    }

    @When("se elimina el cliente con ID {long}")
    public void seEliminaElClienteConID(Long id) {
        try {
            Long idToUse = (inputDTO != null && inputDTO.getId() != null) ? inputDTO.getId() : id;
            customerService.deleteCustomer(idToUse);
        } catch (Exception e) {
            thrownException = e;
        }
    }

    @Then("el cliente es eliminado sin errores")
    public void elClienteEsEliminadoSinErrores() {
        assertNull(thrownException);
        assertFalse(customerRepository.existsById(inputDTO.getId()));
    }
}
