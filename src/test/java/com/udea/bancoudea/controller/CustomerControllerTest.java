package com.udea.bancoudea.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.udea.bancoudea.dto.CustomerDTO;
import com.udea.bancoudea.exception.CustomerNotFoundException;
import com.udea.bancoudea.service.CustomerService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CustomerController.class)
class CustomerControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private CustomerService customerService;

    private CustomerDTO customerDTO;

    @BeforeEach
    void setUp() {
        customerDTO = new CustomerDTO(1L, "Juan", "Perez", "ACC-001", 1000.0);
    }

    @Test
    void getAllCustomers_retornaLista() throws Exception {
        when(customerService.getAllCustomer()).thenReturn(List.of(customerDTO));

        mockMvc.perform(get("/api/customers"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].firstName").value("Juan"))
                .andExpect(jsonPath("$[0].accountNumber").value("ACC-001"));
    }

    @Test
    void getAllCustomers_retornaListaVacia() throws Exception {
        when(customerService.getAllCustomer()).thenReturn(List.of());

        mockMvc.perform(get("/api/customers"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isEmpty());
    }

    @Test
    void getCustomerById_retornaCliente() throws Exception {
        when(customerService.getCustomerById(1L)).thenReturn(customerDTO);

        mockMvc.perform(get("/api/customers/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.firstName").value("Juan"));
    }

    @Test
    void getCustomerById_retorna404CuandoNoExiste() throws Exception {
        when(customerService.getCustomerById(99L)).thenThrow(new CustomerNotFoundException(99L));

        mockMvc.perform(get("/api/customers/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.message").value("Cliente no encontrado con id: 99"));
    }

    @Test
    void createCustomer_retorna201() throws Exception {
        when(customerService.createCustomer(any(CustomerDTO.class))).thenReturn(customerDTO);

        mockMvc.perform(post("/api/customers")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(customerDTO)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.accountNumber").value("ACC-001"));
    }

    @Test
    void createCustomer_retorna400CuandoBalanceNulo() throws Exception {
        CustomerDTO sinBalance = new CustomerDTO(null, "Juan", "Perez", "ACC-001", null);
        when(customerService.createCustomer(any(CustomerDTO.class)))
                .thenThrow(new IllegalArgumentException("Balance cannot be null"));

        mockMvc.perform(post("/api/customers")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(sinBalance)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.message").value("Balance cannot be null"));
    }

    @Test
    void updateCustomer_retornaClienteActualizado() throws Exception {
        CustomerDTO actualizado = new CustomerDTO(1L, "Carlos", "Perez", "ACC-001", 2000.0);
        when(customerService.updateCustomer(eq(1L), any(CustomerDTO.class))).thenReturn(actualizado);

        mockMvc.perform(put("/api/customers/1")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(actualizado)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.firstName").value("Carlos"))
                .andExpect(jsonPath("$.balance").value(2000.0));
    }

    @Test
    void updateCustomer_retorna404CuandoNoExiste() throws Exception {
        when(customerService.updateCustomer(eq(99L), any(CustomerDTO.class)))
                .thenThrow(new CustomerNotFoundException(99L));

        mockMvc.perform(put("/api/customers/99")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(customerDTO)))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }

    @Test
    void deleteCustomer_retorna204() throws Exception {
        doNothing().when(customerService).deleteCustomer(1L);

        mockMvc.perform(delete("/api/customers/1"))
                .andExpect(status().isNoContent());
    }

    @Test
    void deleteCustomer_retorna404CuandoNoExiste() throws Exception {
        doThrow(new CustomerNotFoundException(99L)).when(customerService).deleteCustomer(99L);

        mockMvc.perform(delete("/api/customers/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }
}
