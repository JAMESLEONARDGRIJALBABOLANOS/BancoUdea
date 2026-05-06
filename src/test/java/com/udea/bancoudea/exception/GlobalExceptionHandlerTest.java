package com.udea.bancoudea.exception;

import com.udea.bancoudea.controller.CustomerController;
import com.udea.bancoudea.service.CustomerService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CustomerController.class)
class GlobalExceptionHandlerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private CustomerService customerService;

    @Test
    void debeRetornar404CuandoClienteNoExiste() throws Exception {
        when(customerService.getCustomerById(99L))
                .thenThrow(new CustomerNotFoundException(99L));

        mockMvc.perform(get("/api/customers/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.message").value("Cliente no encontrado con id: 99"));
    }

    @Test
    void debeRetornar400CuandoArgumentoInvalido() throws Exception {
        when(customerService.getCustomerById(99L))
                .thenThrow(new IllegalArgumentException("Argumento invalido"));

        mockMvc.perform(get("/api/customers/99"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.message").value("Argumento invalido"));
    }

    @Test
    void debeRetornar500SinDetallesInternosCuandoErrorGenerico() throws Exception {
        when(customerService.getCustomerById(99L))
                .thenThrow(new RuntimeException("Error interno detallado"));

        mockMvc.perform(get("/api/customers/99"))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.status").value(500))
                .andExpect(jsonPath("$.message").value("Error interno del servidor"));
    }
}
