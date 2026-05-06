package com.udea.bancoudea.service;

import com.udea.bancoudea.dto.CustomerDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.mapper.CustomerMapper;
import com.udea.bancoudea.repository.CustomerRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CustomerServiceTest {

    @Mock
    private CustomerRepository customerRepository;

    @Mock
    private CustomerMapper customerMapper;

    @InjectMocks
    private CustomerService customerService;

    private Customer customer;
    private CustomerDTO customerDTO;

    @BeforeEach
    void setUp() {
        customer = new Customer(1L, "ACC-001", "Juan", "Perez", 1000.0);
        customerDTO = new CustomerDTO(1L, "Juan", "Perez", "ACC-001", 1000.0);
    }

    // --- getAllCustomer ---

    @Test
    void getAllCustomer_retornaListaDeClientes() {
        when(customerRepository.findAll()).thenReturn(List.of(customer));
        when(customerMapper.toDTO(customer)).thenReturn(customerDTO);

        List<CustomerDTO> result = customerService.getAllCustomer();

        assertEquals(1, result.size());
        assertEquals("Juan", result.get(0).getFirstName());
        verify(customerRepository).findAll();
    }

    @Test
    void getAllCustomer_retornaListaVaciaSiNoHayClientes() {
        when(customerRepository.findAll()).thenReturn(List.of());

        List<CustomerDTO> result = customerService.getAllCustomer();

        assertTrue(result.isEmpty());
    }

    // --- getCustomerById ---

    @Test
    void getCustomerById_retornaClienteCuandoExiste() {
        when(customerRepository.findById(1L)).thenReturn(Optional.of(customer));
        when(customerMapper.toDTO(customer)).thenReturn(customerDTO);

        CustomerDTO result = customerService.getCustomerById(1L);

        assertNotNull(result);
        assertEquals("ACC-001", result.getAccountNumber());
    }

    @Test
    void getCustomerById_lanzaExcepcionCuandoNoExiste() {
        when(customerRepository.findById(99L)).thenReturn(Optional.empty());

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> customerService.getCustomerById(99L));

        assertEquals("Cliente no encontrado con id: 99", ex.getMessage());
    }

    // --- createCustomer ---

    @Test
    void createCustomer_guardaYRetornaDTO() {
        when(customerMapper.toEntity(customerDTO)).thenReturn(customer);
        when(customerRepository.save(customer)).thenReturn(customer);
        when(customerMapper.toDTO(customer)).thenReturn(customerDTO);

        CustomerDTO result = customerService.createCustomer(customerDTO);

        assertNotNull(result);
        assertEquals("Juan", result.getFirstName());
        verify(customerRepository).save(customer);
    }

    // --- updateCustomer ---

    @Test
    void updateCustomer_actualizaCamposNoNulos() {
        CustomerDTO actualizacion = new CustomerDTO(null, "Carlos", null, null, 2000.0);

        when(customerRepository.findById(1L)).thenReturn(Optional.of(customer));
        when(customerRepository.save(customer)).thenReturn(customer);
        when(customerMapper.toDTO(customer)).thenReturn(
                new CustomerDTO(1L, "Carlos", "Perez", "ACC-001", 2000.0));

        CustomerDTO result = customerService.updateCustomer(1L, actualizacion);

        assertEquals("Carlos", result.getFirstName());
        assertEquals(2000.0, result.getBalance());
        verify(customerRepository).save(customer);
    }

    @Test
    void updateCustomer_lanzaExcepcionSiClienteNoExiste() {
        when(customerRepository.findById(99L)).thenReturn(Optional.empty());

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> customerService.updateCustomer(99L, customerDTO));

        assertEquals("Cliente no encontrado con id: 99", ex.getMessage());
    }

    // --- deleteCustomer ---

    @Test
    void deleteCustomer_eliminaClienteExistente() {
        when(customerRepository.findById(1L)).thenReturn(Optional.of(customer));

        assertDoesNotThrow(() -> customerService.deleteCustomer(1L));
        verify(customerRepository).deleteById(1L);
    }

    @Test
    void deleteCustomer_lanzaExcepcionSiClienteNoExiste() {
        when(customerRepository.findById(99L)).thenReturn(Optional.empty());

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> customerService.deleteCustomer(99L));

        assertEquals("Cliente no encontrado con id: 99", ex.getMessage());
        verify(customerRepository, never()).deleteById(any());
    }
}
